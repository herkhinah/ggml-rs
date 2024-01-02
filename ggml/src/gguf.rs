use std::{borrow::Borrow, collections::HashSet, marker::PhantomData, path::Path, str::Utf8Error};

use ggml_sys::root::*;
use thiserror::Error;

#[macro_use]
pub use gguf::gguf;

pub use ggml_proc_macro::Deserialize;

pub trait Deserialize<'a>: Sized {
    fn deserialize<B: crate::builder::Builder<'a>>(builder: &mut B) -> Result<Self, B::Error> {
        let (mut res, tensors) = Self::deserialize_relative(String::new(), builder)?;
        builder.alloc(tensors);
        Self::register_tensors(&mut res, String::new(), builder)?;
        Ok(res)
    }

    fn deserialize_relative<B: crate::builder::Builder<'a>>(
        root: String,
        builder: &B,
    ) -> Result<(Self, usize), B::Error>;

    fn register_tensors<B: crate::builder::Builder<'a>>(
        &mut self,
        root: String,
        builder: &mut B,
    ) -> Result<(), B::Error>;
}

#[derive(Error, Debug)]
pub enum GGUFError {
    #[error("expected type {expected:?}, got type {got:?}")]
    TypeMismatch { expected: gguf_type, got: gguf_type },
    #[error("invalid type")]
    InvalidType,
    #[error("invalid utf8 string: {0}")]
    InvalidUtf8String(Utf8Error),
    #[error("{0}")]
    OtherError(String),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("IO error: {0}")]
    IOError(std::io::Error),
}

pub struct GGUFTensorInfo<'a> {
    pub(crate) inner: &'a gguf_tensor_info,
}

impl<'a> std::fmt::Debug for GGUFTensorInfo<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gguf_tensor_info {
            name,
            n_dims,
            ne,
            type_,
            offset,
            data,
            size,
        } = unsafe { &*self.inner };

        let mut res = f.debug_struct("GGUFTensorInfo");
        let res = res.field("name", &self.name());
        match n_dims {
            1 => res.field("dims", &ne[0]),
            2 => res.field("dims", &(ne[0], ne[1])),
            3 => res.field("dims", &(ne[0], ne[1], ne[2])),
            4 => res.field("dims", &(ne[0], ne[1], ne[2], ne[3])),
            _ => res.field("n_dims", &n_dims).field("ne", ne),
        }
        .field("type", type_)
        .field("offset", offset)
        .finish()
    }
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
    inner: &'a gguf_kv,
}

impl<'a> std::fmt::Debug for GGUFKeyValue<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gguf_kv { type_, value, .. } = self.inner;

        let key = self.key();

        let value: Box<dyn std::fmt::Debug> = match type_ {
            gguf_type::UINT8 => Box::new(unsafe { value.uint8 }),
            gguf_type::INT8 => Box::new(unsafe { value.int8 }),
            gguf_type::UINT16 => Box::new(unsafe { value.uint16 }),

            gguf_type::INT16 => Box::new(unsafe { value.int16 }),
            gguf_type::UINT32 => Box::new(unsafe { value.uint32 }),
            gguf_type::INT32 => Box::new(unsafe { value.int32 }),
            gguf_type::FLOAT32 => Box::new(unsafe { value.float32 }),
            gguf_type::BOOL => Box::new(unsafe { value.bool_ }),
            gguf_type::STRING => Box::new(unsafe { value.str_ }),
            gguf_type::ARRAY => Box::new(unsafe { value.arr }),
            gguf_type::UINT64 => Box::new(unsafe { value.uint64 }),
            gguf_type::INT64 => Box::new(unsafe { value.int64 }),
            gguf_type::FLOAT64 => Box::new(unsafe { value.float64 }),
            gguf_type::COUNT => Box::new("invalid type"),
        };

        f.debug_struct("GGUFKeyValue")
            .field("key", &key.to_string())
            .field("type", type_)
            .field("value", &value)
            .finish()
    }
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
    pub tensor_infos: HashSet<GGUFTensorInfo<'a>>,
    pub key_values: HashSet<GGUFKeyValue<'a>>,
}

macro_rules! try_into_gguf_value {
    ($a:ident, $b:ident, $c:ident) => {
        impl<'a> TryInto<$a> for &GGUFKeyValue<'a> {
            type Error = GGUFError;

            fn try_into(self) -> Result<$a, Self::Error> {
                if matches!(self.inner.type_, gguf_type::$b) {
                    Ok(unsafe { self.inner.value.$c })
                } else {
                    Err(GGUFError::TypeMismatch {
                        expected: gguf_type::$b,
                        got: self.inner.type_,
                    })
                }
            }
        }
        impl<'a> TryInto<$a> for GGUFKeyValue<'a> {
            type Error = GGUFError;

            fn try_into(self) -> Result<$a, Self::Error> {
                if matches!(self.inner.type_, gguf_type::$b) {
                    Ok(unsafe { self.inner.value.$c })
                } else {
                    Err(GGUFError::TypeMismatch {
                        expected: gguf_type::$b,
                        got: self.inner.type_,
                    })
                }
            }
        }
    };
}

try_into_gguf_value!(u8, UINT8, uint8);
try_into_gguf_value!(i8, INT8, int8);
try_into_gguf_value!(u16, UINT16, uint16);
try_into_gguf_value!(i16, INT16, int16);
try_into_gguf_value!(u32, UINT32, uint32);
try_into_gguf_value!(i32, INT32, int32);
try_into_gguf_value!(u64, UINT32, uint64);
try_into_gguf_value!(i64, INT32, int64);
try_into_gguf_value!(bool, BOOL, bool_);
try_into_gguf_value!(f32, FLOAT32, float32);
try_into_gguf_value!(f64, FLOAT64, float64);

impl<'a> TryInto<&'a str> for &GGUFKeyValue<'a> {
    type Error = GGUFError;
    fn try_into(self) -> Result<&'a str, Self::Error> {
        if matches!(self.inner.type_, gguf_type::STRING) {
            let gguf_str { n, data } = unsafe { self.inner.value.str_ };
            let slice = unsafe { std::slice::from_raw_parts(data as *const u8, n as usize) };
            Ok(unsafe { std::str::from_utf8_unchecked(slice) })
        } else {
            Err(GGUFError::TypeMismatch {
                expected: gguf_type::STRING,
                got: self.inner.type_,
            })
        }
    }
}
impl<'a> TryInto<String> for &GGUFKeyValue<'a> {
    type Error = GGUFError;
    fn try_into(self) -> Result<String, Self::Error> {
        if matches!(self.inner.type_, gguf_type::STRING) {
            let gguf_str { n, data } = unsafe { self.inner.value.str_ };
            let slice = unsafe { std::slice::from_raw_parts(data as *const u8, n as usize) };
            Ok(unsafe { std::str::from_utf8_unchecked(slice) }.to_string())
        } else {
            Err(GGUFError::TypeMismatch {
                expected: gguf_type::STRING,
                got: self.inner.type_,
            })
        }
    }
}
impl<'a> TryInto<&'a str> for GGUFKeyValue<'a> {
    type Error = GGUFError;
    fn try_into(self) -> Result<&'a str, Self::Error> {
        if matches!(self.inner.type_, gguf_type::STRING) {
            let gguf_str { n, data } = unsafe { self.inner.value.str_ };
            let slice = unsafe { std::slice::from_raw_parts(data as *const u8, n as usize) };
            Ok(unsafe { std::str::from_utf8_unchecked(slice) })
        } else {
            Err(GGUFError::TypeMismatch {
                expected: gguf_type::STRING,
                got: self.inner.type_,
            })
        }
    }
}
impl<'a> TryInto<String> for GGUFKeyValue<'a> {
    type Error = GGUFError;
    fn try_into(self) -> Result<String, Self::Error> {
        if matches!(self.inner.type_, gguf_type::STRING) {
            let gguf_str { n, data } = unsafe { self.inner.value.str_ };
            let slice = unsafe { std::slice::from_raw_parts(data as *const u8, n as usize) };
            Ok(unsafe { std::str::from_utf8_unchecked(slice) }.to_string())
        } else {
            Err(GGUFError::TypeMismatch {
                expected: gguf_type::STRING,
                got: self.inner.type_,
            })
        }
    }
}

macro_rules! try_into_gguf_array {
    ($a:ident, $b:ident, $c:ident) => {
        impl<'a> TryInto<&'a [$a]> for &GGUFKeyValue<'a> {
            type Error = GGUFError;

            fn try_into(self) -> Result<&'a [$a], Self::Error> {
                if matches!(self.inner.type_, gguf_type::ARRAY) {
                    unsafe {
                        let gguf_value__bindgen_ty_1 { type_, n, data } = self.inner.value.arr;

                        if matches!(type_, gguf_type::$b) {
                            Ok(std::slice::from_raw_parts(data as *const $a, n as usize))
                        } else {
                            Err(GGUFError::TypeMismatch {
                                expected: gguf_type::$b,
                                got: type_,
                            })
                        }
                    }
                } else {
                    Err(GGUFError::TypeMismatch {
                        expected: gguf_type::ARRAY,
                        got: self.inner.type_,
                    })
                }
            }
        }
        impl<'a> TryInto<&'a [$a]> for GGUFKeyValue<'a> {
            type Error = GGUFError;

            fn try_into(self) -> Result<&'a [$a], Self::Error> {
                if matches!(self.inner.type_, gguf_type::ARRAY) {
                    unsafe {
                        let gguf_value__bindgen_ty_1 { type_, n, data } = self.inner.value.arr;

                        if matches!(type_, gguf_type::$b) {
                            Ok(std::slice::from_raw_parts(data as *const $a, n as usize))
                        } else {
                            Err(GGUFError::TypeMismatch {
                                expected: gguf_type::$b,
                                got: type_,
                            })
                        }
                    }
                } else {
                    Err(GGUFError::TypeMismatch {
                        expected: gguf_type::ARRAY,
                        got: self.inner.type_,
                    })
                }
            }
        }
        impl<'a> TryInto<Vec<$a>> for &GGUFKeyValue<'a> {
            type Error = GGUFError;

            fn try_into(self) -> Result<Vec<$a>, Self::Error> {
                let slice: &[$a] = self.try_into()?;
                Ok(slice.to_vec())
            }
        }
        impl<'a> TryInto<Vec<$a>> for GGUFKeyValue<'a> {
            type Error = GGUFError;

            fn try_into(self) -> Result<Vec<$a>, Self::Error> {
                let slice: &[$a] = self.try_into()?;
                Ok(slice.to_vec())
            }
        }
    };
}

try_into_gguf_array!(u8, UINT8, uint8);
try_into_gguf_array!(i8, INT8, int8);
try_into_gguf_array!(u16, UINT16, uint16);
try_into_gguf_array!(i16, INT16, int16);
try_into_gguf_array!(u32, UINT32, uint32);
try_into_gguf_array!(i32, INT32, int32);
try_into_gguf_array!(u64, UINT32, uint64);
try_into_gguf_array!(i64, INT32, int64);
try_into_gguf_array!(bool, BOOL, bool_);
try_into_gguf_array!(f32, FLOAT32, float32);
try_into_gguf_array!(f64, FLOAT64, float64);

impl<'a> TryInto<Vec<&'a str>> for &GGUFKeyValue<'a> {
    type Error = GGUFError;

    fn try_into(self) -> Result<Vec<&'a str>, Self::Error> {
        let gguf_kv { type_, value, .. } = self.inner;

        if matches!(type_, gguf_type::ARRAY) {
            let gguf_value__bindgen_ty_1 { type_, n, data } = unsafe { value.arr };

            if matches!(type_, gguf_type::STRING) {
                let slice =
                    unsafe { std::slice::from_raw_parts(data as *const gguf_str, n as usize) };

                let mut vec = Vec::with_capacity(n as usize);

                for gguf_str { n, data } in slice {
                    let slice =
                        unsafe { std::slice::from_raw_parts(*data as *const u8, *n as usize) };
                    vec.push(unsafe { std::str::from_utf8_unchecked(slice) });
                }

                Ok(vec)
            } else {
                Err(GGUFError::TypeMismatch {
                    expected: gguf_type::STRING,
                    got: type_,
                })
            }
        } else {
            Err(GGUFError::TypeMismatch {
                expected: gguf_type::ARRAY,
                got: type_.clone(),
            })
        }
    }
}
impl<'a> TryInto<Vec<String>> for &GGUFKeyValue<'a> {
    type Error = GGUFError;

    fn try_into(self) -> Result<Vec<String>, Self::Error> {
        let gguf_kv { type_, value, .. } = self.inner;

        if matches!(type_, gguf_type::ARRAY) {
            let gguf_value__bindgen_ty_1 { type_, n, data } = unsafe { value.arr };

            if matches!(type_, gguf_type::STRING) {
                let slice =
                    unsafe { std::slice::from_raw_parts(data as *const gguf_str, n as usize) };

                let mut vec = Vec::with_capacity(n as usize);

                for gguf_str { n, data } in slice {
                    let slice =
                        unsafe { std::slice::from_raw_parts(*data as *const u8, *n as usize) };
                    vec.push(unsafe { std::str::from_utf8_unchecked(slice) }.to_string());
                }

                Ok(vec)
            } else {
                Err(GGUFError::TypeMismatch {
                    expected: gguf_type::STRING,
                    got: type_,
                })
            }
        } else {
            Err(GGUFError::TypeMismatch {
                expected: gguf_type::ARRAY,
                got: type_.clone(),
            })
        }
    }
}
impl<'a> TryInto<Vec<&'a str>> for GGUFKeyValue<'a> {
    type Error = GGUFError;

    fn try_into(self) -> Result<Vec<&'a str>, Self::Error> {
        let Self {
            inner: gguf_kv { type_, value, .. },
        } = self;

        if matches!(type_, gguf_type::ARRAY) {
            let gguf_value__bindgen_ty_1 { type_, n, data } = unsafe { value.arr };

            if matches!(type_, gguf_type::STRING) {
                let slice =
                    unsafe { std::slice::from_raw_parts(data as *const gguf_str, n as usize) };

                let mut vec = Vec::with_capacity(n as usize);

                for gguf_str { n, data } in slice {
                    let slice =
                        unsafe { std::slice::from_raw_parts(*data as *const u8, *n as usize) };
                    vec.push(unsafe { std::str::from_utf8_unchecked(slice) });
                }

                Ok(vec)
            } else {
                Err(GGUFError::TypeMismatch {
                    expected: gguf_type::STRING,
                    got: type_,
                })
            }
        } else {
            Err(GGUFError::TypeMismatch {
                expected: gguf_type::ARRAY,
                got: type_.clone(),
            })
        }
    }
}
impl<'a> TryInto<Vec<String>> for GGUFKeyValue<'a> {
    type Error = GGUFError;

    fn try_into(self) -> Result<Vec<String>, Self::Error> {
        let Self {
            inner: gguf_kv { type_, value, .. },
        } = self;

        if matches!(type_, gguf_type::ARRAY) {
            let gguf_value__bindgen_ty_1 { type_, n, data } = unsafe { value.arr };

            if matches!(type_, gguf_type::STRING) {
                let slice =
                    unsafe { std::slice::from_raw_parts(data as *const gguf_str, n as usize) };

                let mut vec = Vec::with_capacity(n as usize);

                for gguf_str { n, data } in slice {
                    let slice =
                        unsafe { std::slice::from_raw_parts(*data as *const u8, *n as usize) };
                    vec.push(unsafe { std::str::from_utf8_unchecked(slice) }.to_string());
                }

                Ok(vec)
            } else {
                Err(GGUFError::TypeMismatch {
                    expected: gguf_type::STRING,
                    got: type_,
                })
            }
        } else {
            Err(GGUFError::TypeMismatch {
                expected: gguf_type::ARRAY,
                got: type_.clone(),
            })
        }
    }
}

impl<'a> GGUFContext<'a> {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, GGUFError> {
        let path_bytes = {
            let bytes = path.as_ref().to_string_lossy();
            let len = bytes.as_bytes().len();
            let mut path = std::iter::repeat(0u8).take(len + 1).collect::<Vec<_>>();
            path.as_mut_slice()[0..len].copy_from_slice(bytes.as_bytes());
            path.push(0u8);
            path
        };

        let params = gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut::<*mut ggml_context>(),
        };

        let gguf_ctx = unsafe { gguf_init_from_file(path_bytes.as_ptr() as *const i8, params) };
        if gguf_ctx.is_null() {
            return Err(GGUFError::OtherError(format!(
                "failed to load gguf file: {}",
                path.as_ref().to_path_buf().to_string_lossy()
            )));
        }

        let (n_kv, n_tensors) = unsafe { ((*gguf_ctx).header.n_kv, (*gguf_ctx).header.n_tensors) };

        let kv = unsafe {
            std::slice::from_raw_parts((*gguf_ctx).kv, n_kv as usize)
            //std::mem::transmute::<&[gguf_kv], &[GGUFKeyValue]>(kv)
        };

        for kv in kv {
            let _ = unsafe { check_gguf_str(kv.key).map_err(GGUFError::InvalidUtf8String)? };
        }

        let key_values: HashSet<_> = kv.iter().map(|kv| GGUFKeyValue { inner: kv }).collect();

        let tensor_infos =
            unsafe { std::slice::from_raw_parts((*gguf_ctx).infos, n_tensors as usize) };

        let tensor_infos: HashSet<_> = tensor_infos
            .iter()
            .map(|t| GGUFTensorInfo { inner: t })
            .collect();

        Ok(GGUFContext {
            gguf_ctx,
            key_values,
            tensor_infos,
        })
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
        }
    }
}
