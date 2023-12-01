use std::{
    borrow::Borrow,
    collections::HashSet,
    ffi::{c_uint, CStr},
    marker::PhantomData,
    os::raw::c_void,
    path::Path,
    str::Utf8Error,
};

use ggml_sys::root::*;
use thiserror::Error;

use crate::TensorsIterator;

pub use ggml_proc_macro::Deserialize;

pub trait Deserialize<'a>: Sized {
    fn deserialize<B: crate::builder::Builder<'a>>(builder: &mut B) -> Result<Self, B::Error> {
        let (res, tensors) = Self::deserialize_relative(String::new(), builder)?;
        builder.alloc(tensors);
        ghost_cell::GhostToken::new(|mut token| {
            let res = ghost_cell::GhostCell::new(res);

            Self::register_tensors(&res, &mut token, String::new(), builder)?;
            Ok(res.into_inner())
        })
    }

    fn deserialize_relative<B: crate::builder::Builder<'a>>(
        root: String,
        builder: &B,
    ) -> Result<(Self, usize), B::Error>;

    fn register_tensors<'id, B: crate::builder::Builder<'a>>(
        this: &ghost_cell::GhostCell<'id, Self>,
        token: &mut ghost_cell::GhostToken<'id>,
        root: String,
        builder: &mut B,
    ) -> Result<(), B::Error>;
}

#[derive(Error, Debug)]
pub enum GGUFError {
    #[error("expected type {expected}, got type {got}")]
    TypeMismatch { expected: GGUFType, got: GGUFType },
    #[error("invalid type")]
    InvalidType,
    #[error("invalid utf8 string: {0}")]
    InvalidUtf8String(Utf8Error),
    #[error("{0}")]
    OtherError(String),
    #[error("IO error: {0}")]
    IOError(std::io::Error),
}

pub struct GGUFTensorInfo<'a> {
    pub(crate) inner: &'a gguf_tensor_info,
}

impl<'a> PartialEq for GGUFTensorInfo<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.name == other.inner.name
    }
}

impl<'a> Eq for GGUFTensorInfo<'a> {}

impl<'a> std::hash::Hash for GGUFTensorInfo<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let gguf_str { n, data } = self.inner.name;
        let c_str = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(data as *mut u8, n as usize))
        };

        c_str.hash(state);
    }
}

impl<'a> GGUFTensorInfo<'a> {
    pub fn name(&self) -> &'a str {
        let gguf_str { n, data } = self.inner.name;
        let slice = unsafe { std::slice::from_raw_parts(data as *mut u8, n as usize) };
        let slice = unsafe { std::str::from_utf8_unchecked(slice) };
        slice
    }
}

impl<'a> Borrow<str> for GGUFTensorInfo<'a> {
    fn borrow(&self) -> &'a str {
        let gguf_str { n, data } = self.inner.name;
        let slice = unsafe { std::slice::from_raw_parts(data as *mut u8, n as usize) };
        let slice = unsafe { std::str::from_utf8_unchecked(slice) };
        slice
    }
}

pub struct GGUFKeyValue<'a> {
    inner: gguf_kv,
    phantom: PhantomData<&'a GGUFContext<'a>>,
}

impl<'a> GGUFKeyValue<'a> {
    pub fn key(&self) -> &str {
        let gguf_str { n, data } = self.inner.key;
        let slice = unsafe { std::slice::from_raw_parts(data as *mut u8, n as usize) };
        let slice = unsafe { std::str::from_utf8_unchecked(slice) };
        slice
    }
}

impl<'a> Borrow<str> for GGUFKeyValue<'a> {
    fn borrow(&self) -> &str {
        let gguf_str { n, data } = self.inner.key;
        let slice = unsafe { std::slice::from_raw_parts(data as *mut u8, n as usize) };
        let slice = unsafe { std::str::from_utf8_unchecked(slice) };
        slice
    }
}

impl<'a> PartialEq for GGUFKeyValue<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.key == other.inner.key
    }
}

impl<'a> Eq for GGUFKeyValue<'a> {}

impl<'a> std::hash::Hash for GGUFKeyValue<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let gguf_str { n, data } = self.inner.key;
        let c_str = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(data as *mut u8, n as usize))
        };

        c_str.hash(state);
    }
}

pub struct GGUFContext<'a> {
    pub(crate) gguf_ctx: *mut gguf_context,
    pub(crate) ggml_ctx: *mut ggml_context,
    tensors: HashSet<GGUFTensorInfo<'a>>,
}

macro_rules! try_into_gguf_value {
    ($a:ident, $b:ident, $c:ident) => {
        impl<'a> TryInto<$a> for GGUFKeyValue<'a> {
            type Error = GGUFError;

            fn try_into(self) -> Result<$a, Self::Error> {
                if GGUFType::into_discriminant(GGUFType::$b) == self.inner.type_ {
                    Ok(unsafe { self.inner.value.$c })
                } else {
                    Err(GGUFError::TypeMismatch {
                        expected: GGUFType::$b,
                        got: GGUFType::from_discriminant(self.inner.type_).unwrap(),
                    })
                }
            }
        }
    };
}

try_into_gguf_value!(u8, UInt8, uint8);
try_into_gguf_value!(i8, Int8, int8);
try_into_gguf_value!(u16, UInt16, uint16);
try_into_gguf_value!(i16, Int16, int16);
try_into_gguf_value!(u32, UInt32, uint32);
try_into_gguf_value!(i32, Int32, int32);
try_into_gguf_value!(u64, UInt32, uint64);
try_into_gguf_value!(i64, Int32, int64);
try_into_gguf_value!(bool, Bool, bool_);
try_into_gguf_value!(f32, Float32, float32);
try_into_gguf_value!(f64, Float64, float64);

impl<'a> TryInto<&'a str> for GGUFKeyValue<'a> {
    type Error = GGUFError;
    fn try_into(self) -> Result<&'a str, Self::Error> {
        if GGUFType::String.into_discriminant() == self.inner.type_ {
            let gguf_str { n, data } = unsafe { self.inner.value.str_ };
            let slice = unsafe { std::slice::from_raw_parts(data as *const u8, n as usize) };
            Ok(unsafe { std::str::from_utf8_unchecked(slice) })
        } else {
            Err(GGUFError::TypeMismatch {
                expected: GGUFType::String,
                got: GGUFType::from_discriminant(self.inner.type_).unwrap(),
            })
        }
    }
}

macro_rules! try_into_gguf_array {
    ($a:ident, $b:ident, $c:ident) => {
        impl<'a> TryInto<&'a [$a]> for GGUFKeyValue<'a> {
            type Error = GGUFError;

            fn try_into(self) -> Result<&'a [$a], Self::Error> {
                if GGUFType::Array.into_discriminant() == self.inner.type_ {
                    unsafe {
                        let gguf_value__bindgen_ty_1 { type_, n, data } = self.inner.value.arr;

                        if let GGUFType::$b = GGUFType::from_discriminant(type_)
                            .map_err(|_| GGUFError::InvalidType)?
                        {
                            Ok(std::slice::from_raw_parts(data as *const $a, n as usize))
                        } else {
                            Err(GGUFError::TypeMismatch {
                                expected: GGUFType::$b,
                                got: GGUFType::from_discriminant(type_)
                                    .map_err(|_| GGUFError::InvalidType)?,
                            })
                        }
                    }
                } else {
                    Err(GGUFError::TypeMismatch {
                        expected: GGUFType::Array,
                        got: GGUFType::from_discriminant(self.inner.type_).unwrap(),
                    })
                }
            }
        }
    };
}

try_into_gguf_array!(u8, UInt8, uint8);
try_into_gguf_array!(i8, Int8, int8);
try_into_gguf_array!(u16, UInt16, uint16);
try_into_gguf_array!(i16, Int16, int16);
try_into_gguf_array!(u32, UInt32, uint32);
try_into_gguf_array!(i32, Int32, int32);
try_into_gguf_array!(u64, UInt32, uint64);
try_into_gguf_array!(i64, Int32, int64);
try_into_gguf_array!(bool, Bool, bool_);
try_into_gguf_array!(f32, Float32, float32);
try_into_gguf_array!(f64, Float64, float64);

#[derive(Copy, Clone)]
#[repr(C)]
struct GGUFArr {
    gguf_type: c_uint,
    n: u64,
    data: *const c_void,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct GGUFStr {
    n: u64,
    data: *const i8,
}

#[derive(Clone, Copy, Debug)]
#[repr(u32)]

pub enum GGUFType {
    UInt8 = gguf_type_GGUF_TYPE_UINT8,
    Int8 = gguf_type_GGUF_TYPE_INT8,
    UInt16 = gguf_type_GGUF_TYPE_UINT16,
    Int16 = gguf_type_GGUF_TYPE_INT16,
    UInt32 = gguf_type_GGUF_TYPE_UINT32,
    Int32 = gguf_type_GGUF_TYPE_INT32,
    UInt64 = gguf_type_GGUF_TYPE_UINT64,
    Int64 = gguf_type_GGUF_TYPE_INT64,
    Float32 = gguf_type_GGUF_TYPE_FLOAT32,
    Float64 = gguf_type_GGUF_TYPE_FLOAT64,
    Bool = gguf_type_GGUF_TYPE_BOOL,
    String = gguf_type_GGUF_TYPE_STRING,
    Array = gguf_type_GGUF_TYPE_ARRAY,
    Count = gguf_type_GGUF_TYPE_COUNT,
}

impl std::fmt::Display for GGUFType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGUFType::UInt8 => f.write_str("u8"),
            GGUFType::Int8 => f.write_str("i8"),
            GGUFType::UInt16 => f.write_str("u16"),
            GGUFType::Int16 => f.write_str("i32"),
            GGUFType::UInt32 => f.write_str("u32"),
            GGUFType::Int32 => f.write_str("i32"),
            GGUFType::UInt64 => f.write_str("u64"),
            GGUFType::Int64 => f.write_str("i64"),
            GGUFType::Float32 => f.write_str("f32"),
            GGUFType::Float64 => f.write_str("f64"),
            GGUFType::Bool => f.write_str("bool"),
            GGUFType::String => f.write_str("string"),
            GGUFType::Array => f.write_str("array"),
            GGUFType::Count => f.write_str("invalid type"),
        }
    }
}

impl GGUFType {
    fn into_discriminant(self) -> c_uint {
        // SAFETY: Because `Self` is marked `repr(u8)`, its layout is a `repr(C)` `union`
        // between `repr(C)` structs, each of which has the `u8` discriminant as its first
        // field, so we can read the discriminant without offsetting the pointer.
        unsafe { std::mem::transmute::<GGUFType, c_uint>(self) }
    }

    fn from_discriminant(v: c_uint) -> Result<GGUFType, GGUFError> {
        if v < GGUFType::Count.into_discriminant() {
            Ok(unsafe { std::mem::transmute::<c_uint, GGUFType>(v) })
        } else {
            Err(GGUFError::InvalidType)
        }
    }
}
pub struct GGUFKey(i32);

impl GGUFType {
    pub fn as_cstr(&self) -> &CStr {
        unsafe { CStr::from_ptr(gguf_type_name(*self as u32)) }
    }
}

impl<'a> GGUFContext<'a> {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, GGUFError> {
        let mut ggml_ctx = std::pin::pin!(std::ptr::null_mut::<ggml_context>());

        let params = gguf_init_params {
            no_alloc: true,
            ctx: unsafe { ggml_ctx.as_mut().get_unchecked_mut() as *mut *mut ggml_context },
        };

        let path_bytes = {
            let bytes = path.as_ref().to_string_lossy();
            let len = bytes.as_bytes().len();
            let mut path = std::iter::repeat(0u8).take(len + 1).collect::<Vec<_>>();
            path.as_mut_slice()[0..len].copy_from_slice(bytes.as_bytes());
            path.push(0u8);
            path
        };

        let gguf_ctx = unsafe { gguf_init_from_file(path_bytes.as_ptr() as *const i8, params) };
        if gguf_ctx.is_null() {
            return Err(GGUFError::OtherError(format!(
                "failed to load gguf file: {}",
                path.as_ref().to_path_buf().to_string_lossy()
            )));
        }

        let (n_kv, n_tensors) = unsafe { ((*gguf_ctx).header.n_kv, (*gguf_ctx).header.n_tensors) };

        let mut res = GGUFContext {
            gguf_ctx,
            ggml_ctx: *std::pin::Pin::into_inner(ggml_ctx),
            tensors: HashSet::with_capacity(n_tensors as usize),
        };

        let kv = unsafe {
            let kv = std::slice::from_raw_parts((*gguf_ctx).kv, n_kv as usize);
            std::mem::transmute::<&[gguf_kv], &[GGUFKeyValue]>(kv)
        };

        for kv in kv {
            let _ = unsafe { check_gguf_str(kv.inner.key).map_err(GGUFError::InvalidUtf8String)? };
        }

        let tensor_infos =
            unsafe { std::slice::from_raw_parts((*gguf_ctx).infos, n_tensors as usize) };

        for tensor_info in tensor_infos.iter() {
            res.tensors.insert(GGUFTensorInfo { inner: tensor_info });
        }

        Ok(res)
    }

    pub fn tensor_iter<'b>(&'b self) -> TensorsIterator<'b> {
        unsafe { TensorsIterator::new(self.ggml_ctx) }
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GGUFTensorInfo<'a>> {
        self.tensors.get(name)
    }

    /*
    pub fn tensors(&self) -> &[GGUFTensorInfo<'a>] {
        unsafe {
            let tensors = std::slice::from_raw_parts(
                (*self.gguf_ctx).infos,
                (*self.gguf_ctx).header.n_tensors as usize,
            );
            std::mem::transmute::<&[gguf_tensor_info], &[GGUFTensorInfo]>(tensors)
        }
    }*/

    pub fn kv(&self) -> &[GGUFKeyValue<'_>] {
        unsafe {
            let gguf_context { header, kv, .. } = &*self.gguf_ctx;
            let kv = std::slice::from_raw_parts(*kv, header.n_kv as usize);
            std::mem::transmute::<&[gguf_kv], &[GGUFKeyValue]>(kv)
        }
    }
}

unsafe fn check_gguf_str<'a>(str: gguf_str) -> Result<&'a str, Utf8Error> {
    let gguf_str { n, data } = str;
    let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, n as usize) };
    std::str::from_utf8(bytes)
}

impl<'a> Drop for GGUFContext<'a> {
    fn drop(&mut self) {
        unsafe {
            gguf_free(self.gguf_ctx);
            ggml_free(self.ggml_ctx);
        }
    }
}
