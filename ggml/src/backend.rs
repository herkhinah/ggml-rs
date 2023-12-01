use ggml_sys::root::*;

use crate::{error::GError, tensor::GTensor};

#[cfg(feature = "cublas")]
pub mod cublas;

pub trait GBackend<'a> {
    unsafe fn backend(&'a self) -> *mut ggml_backend;
    fn tensor(&'a self, name: &str) -> &GTensor<'a>;
}

pub struct CpuBackend(*mut ggml_backend);

impl CpuBackend {
    pub fn new() -> Result<Self, GError> {
        let backend = unsafe { ggml_backend_cpu_init() };
        if backend.is_null() {
            return Err(GError::CpuBackendInitFailed);
        }
        Ok(CpuBackend(backend))
    }
}

/*
impl GBackend for CpuBackend {
    fn backend(&self) -> *mut ggml_backend {
        self.0
    }
}

impl Deref for CpuBackend {
    type Target = GBackend;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
*/
