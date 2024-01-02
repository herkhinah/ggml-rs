use std::marker::PhantomData;

use ggml_sys::root::*;

use crate::{error::Error, tensor::Tensor};

#[cfg(feature = "cublas")]
pub mod cublas;

#[deprecated]
pub trait GBackend<'a> {
    unsafe fn backend(&'a self) -> *mut ggml_backend;
    fn tensor(&'a self, name: &str) -> &Tensor<'a>;
}

#[repr(transparent)]
pub struct Backend(*mut ggml_backend);

#[repr(transparent)]
pub struct CpuBackend(Backend);

impl CpuBackend {
    pub fn new() -> Result<Self, Error> {
        let backend = unsafe { ggml_backend_cpu_init() };
        if backend.is_null() {
            return Err(Error::CpuBackendInitFailed);
        }
        Ok(CpuBackend(Backend(backend)))
    }

    pub fn set_threads(&mut self, threads: u16) {
        unsafe {
            ggml_backend_cpu_set_n_threads(self.0 .0, threads as i32);
        }
    }
}

impl Into<Backend> for CpuBackend {
    fn into(self) -> Backend {
        self.0
    }
}

impl TryInto<CpuBackend> for Backend {
    type Error = crate::error::Error;

    fn try_into(self) -> Result<CpuBackend, Self::Error> {
        unsafe {
            if ggml_backend_is_cpu(self.0) {
                return Ok(CpuBackend(self));
            }
            return Err(Error::BackendNotCpu);
        }
    }
}

impl Backend {
    pub fn cpu_backend(threads: u16) -> Result<Self, Error> {
        let mut backend = CpuBackend::new()?;
        backend.set_threads(threads);

        Ok(backend.into())
    }

    pub fn gpu_backend() -> Self {
        Self(unsafe { ggml_backend_cuda_init(0) })
    }
}

pub struct BackendBuffer<'a> {
    inner: *mut ggml_backend_buffer,
    _marker: PhantomData<&'a Backend>,
}

impl<'a> BackendBuffer<'a> {
    pub fn new(backend: &mut Backend, size: usize) -> Self {
        unsafe {
            Self {
                inner: ggml_backend_alloc_buffer(backend.0, size),
                _marker: PhantomData,
            }
        }
    }
}
