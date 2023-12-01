use std::{marker::PhantomData, ptr::null_mut};

use ggml_sys::root::{ggml_tensor, ggml_type, ggml_type_GGML_TYPE_F16, ggml_type_GGML_TYPE_F32};

use super::GGMLContext;

pub trait DType {
    const GGML_TYPE: ggml_type;
}

pub struct F32();

impl DType for F32 {
    const GGML_TYPE: ggml_type = ggml_type_GGML_TYPE_F32;
}

pub struct F16();

impl DType for F16 {
    const GGML_TYPE: ggml_type = ggml_type_GGML_TYPE_F16;
}

pub struct GTensor<'a>(
    pub(crate) *mut ggml_tensor,
    pub(crate) PhantomData<&'a GGMLContext>,
);

//impl<'a> Unpin! for GTensor<'a>;

impl<'a> GTensor<'a> {
    pub unsafe fn null() -> Self {
        Self(null_mut(), PhantomData)
    }

    pub fn name(&self) -> &str {
        std::ffi::CStr::from_bytes_until_nul(unsafe {
            std::mem::transmute((*self.0).name.as_slice())
        })
        .unwrap()
        .to_str()
        .unwrap()
    }

    /*
    pub fn reshape_1d(&self, ctx: &'a GGMLContext, dim: usize) -> GTensor<'a> {
        unsafe { Self(ggml_reshape_1d(ctx.0, self.0, dim as i64), PhantomData) }
    }

    pub fn tensor_2d(&self, ctx: &'a GGMLContext, dim_x: usize, dim_y: usize) -> GTensor<'a, T> {
        GTensor::<T>(
            unsafe { ggml_reshape_2d(ctx.0, self.0, dim_x as i64, dim_y as i64) },
            PhantomData,
        )
    }

    pub fn tensor_3d(
        &self,
        ctx: &'a GGMLContext,
        dim_x: usize,
        dim_y: usize,
        dim_z: usize,
    ) -> GTensor<'a, T> {
        GTensor::<T>(
            unsafe { ggml_reshape_3d(ctx.0, self.0, dim_x as i64, dim_y as i64, dim_z as i64) },
            PhantomData,
        )
    }

    pub fn tensor_4d(
        &self,
        ctx: &'a GGMLContext,
        dim_x: usize,
        dim_y: usize,
        dim_z: usize,
        dim_w: usize,
    ) -> GTensor<'a, T> {
        GTensor::<T>(
            unsafe {
                ggml_reshape_4d(
                    ctx.0,
                    self.0,
                    dim_x as i64,
                    dim_y as i64,
                    dim_z as i64,
                    dim_w as i64,
                )
            },
            PhantomData,
        )
    }*/
}
