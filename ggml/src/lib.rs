use std::marker::PhantomData;

use ggml_sys::root::*;

use crate::tensor::GTensor;

pub use ggml_sys::root::ggml_type;

pub mod backend;
pub mod builder;
pub mod error;
pub mod gguf;
pub mod tensor;

pub struct GGMLContext(pub(crate) *mut ggml_context);

pub struct TensorsIterator<'a> {
    next_tensor: *mut ggml_tensor,
    ggml_ctx: *mut ggml_context,
    phantom: PhantomData<&'a GGMLContext>,
}

impl<'a> TensorsIterator<'a> {
    unsafe fn new(ggml_ctx: *mut ggml_context) -> Self {
        let next_tensor = unsafe { ggml_get_first_tensor(ggml_ctx) };

        TensorsIterator {
            next_tensor,
            ggml_ctx,
            phantom: PhantomData,
        }
    }
}

impl<'a> Iterator for TensorsIterator<'a> {
    type Item = GTensor<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_tensor.is_null() {
            return None;
        }

        let result = GTensor(self.next_tensor, PhantomData);
        self.next_tensor = unsafe { ggml_get_next_tensor(self.ggml_ctx, self.next_tensor) };
        Some(result)
    }
}

impl GGMLContext {
    pub fn new() -> Self {
        let init_params = ggml_init_params {
            mem_size: 0,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };

        let ctx = unsafe { ggml_init(init_params) };
        Self(ctx)
    }

    pub fn tensor_iter<'a>(&'a self) -> TensorsIterator<'a> {
        unsafe { TensorsIterator::new(self.0) }
    }
}

impl<'a> Drop for GGMLContext {
    fn drop(&mut self) {
        unsafe { ggml_free(self.0) }
    }
}
