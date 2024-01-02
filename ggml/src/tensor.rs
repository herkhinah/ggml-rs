use std::{marker::PhantomData, ptr::null_mut};

use ggml_sys::root::{ggml_new_tensor, ggml_set_name, ggml_tensor};

use crate::{error::Error, gguf::GGUFTensorInfo};

use super::Context;

pub struct Tensor<'a>(
    pub(crate) *mut ggml_tensor,
    pub(crate) PhantomData<&'a Context>,
);

impl<'a> Tensor<'a> {
    pub unsafe fn null() -> Self {
        Self(null_mut(), PhantomData)
    }

    pub unsafe fn from_tensor_info(
        ctx: &'a Context,
        info: &GGUFTensorInfo<'_>,
    ) -> Result<Self, Error> {
        let ggml_tensor = unsafe {
            let ne = [
                (*info.inner).ne[0] as i64,
                (*info.inner).ne[1] as i64,
                (*info.inner).ne[2] as i64,
                (*info.inner).ne[3] as i64,
            ];

            ggml_new_tensor(
                ctx.0,
                (*info.inner).type_,
                (*info.inner).n_dims as i32,
                ne.as_ptr(),
            )
        };

        if ggml_tensor.is_null() {
            return Err(Error::GGMLNewTensorFailed);
        }

        unsafe {
            ggml_set_name(ggml_tensor, (*info.inner).name.data);
        }

        Ok(Self(ggml_tensor, PhantomData))
    }

    pub fn name(&self) -> &str {
        assert_ne!(self.0, null_mut());

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

impl<'a> std::fmt::Debug for Tensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GTensor").finish_non_exhaustive()
    }
}
