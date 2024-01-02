use std::{
    io::{Read, Seek, SeekFrom},
    marker::PhantomData,
    path::Path,
    string::FromUtf8Error,
};

use ggml_sys::root::*;
use thiserror::Error;

use crate::{
    builder::Builder,
    error::Error,
    gguf::{GGUFContext, GGUFError, GGUFKeyValue, GGUFTensorInfo},
    tensor::Tensor,
    Context,
};

pub struct CudaBackend {
    _ggml_backend: *mut ggml_backend,
    _ggml_context: *mut ggml_context,
}

pub struct CudaDevice(std::ffi::c_int);

impl CudaDevice {
    pub fn iter() -> CudaDeviceIter {
        CudaDeviceIter::new()
    }

    pub fn info(&self) -> Result<String, FromUtf8Error> {
        let mut buffer: Vec<u8> = Vec::with_capacity(1024);

        // TODO: invalid utf-8 bytes when description buffer is too short?
        unsafe {
            ggml_cuda_get_device_description(
                self.0,
                buffer.as_mut_ptr() as *mut i8,
                buffer.capacity(),
            )
        };
        String::from_utf8(buffer)
    }
}

pub struct CudaDeviceIter {
    ix: std::ffi::c_int,
    length: std::ffi::c_int,
}

impl CudaDeviceIter {
    fn new() -> Self {
        Self {
            ix: 0,
            length: unsafe { ggml_cuda_get_device_count() },
        }
    }
}

impl Iterator for CudaDeviceIter {
    type Item = CudaDevice;

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = None;
        if self.ix < self.length {
            result = Some(CudaDevice(self.ix));
            self.ix += 1;
        }
        result
    }
}

pub struct CudaBackendBuilder<'a> {
    pub(crate) _backend: *mut ggml_backend,
    pub(crate) gguf_ctx: GGUFContext<'a>,
    pub(crate) gguf_file: std::fs::File,
    pub(crate) ggml_context: Option<Context>,
}

impl<'a> CudaBackendBuilder<'a> {
    pub fn from_gguf<P: AsRef<Path>>(gguf_path: P, device: CudaDevice) -> Result<Self, GGUFError> {
        let gguf_ctx = GGUFContext::from_file(&gguf_path)?;
        let gguf_file = std::fs::File::open(gguf_path).map_err(GGUFError::IOError)?;

        let _backend = unsafe { ggml_backend_cuda_init(device.0) };
        Ok(CudaBackendBuilder {
            _backend,
            gguf_ctx,
            gguf_file,
            ggml_context: None,
        })
    }
}

#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("tensor with this name already exists")]
    DuplicateTensor,
    #[error("{0}")]
    GGUFError(#[from] GGUFError),
    #[error("{0}")]
    GError(#[from] Error),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("key not found: {0}")]
    KeyValueNotFound(String),
}

impl<'a> Builder<'a> for CudaBackendBuilder<'a> {
    type Error = BuilderError;

    /*
    fn tensors(&self) -> TensorsIterator<'_> {
        self.gguf_ctx.tensor_iter()
    }
    */
    fn get_tensor_info(&self, name: &str) -> Result<&GGUFTensorInfo<'a>, Self::Error> {
        match self.gguf_ctx.tensor_infos.get(name) {
            Some(d) => Ok(d),
            None => Err(BuilderError::TensorNotFound(name.to_string())),
        }
    }

    fn alloc(&mut self, tensors: usize) {
        let mem_size = tensors * (GGML_OBJECT_SIZE + GGML_TENSOR_SIZE);
        let mem_buffer = std::ptr::null_mut();
        let no_alloc = true;

        let init_params = ggml_init_params {
            mem_size,
            mem_buffer,
            no_alloc,
        };

        let ctx = Context(unsafe { ggml_init(init_params) });

        self.ggml_context = Some(ctx);
    }

    fn load_tensor(&mut self, name: &str) -> Result<Tensor<'a>, Self::Error> {
        let Self {
            gguf_ctx,
            gguf_file,
            ggml_context: Some(ctx),
            ..
        } = self
        else {
            panic!()
        };
        let data_offset = unsafe { gguf_get_data_offset(gguf_ctx.gguf_ctx) } as u64;

        let Some(tensor_info) = self.gguf_ctx.tensor_infos.get(name) else {
            return Err(BuilderError::TensorNotFound(name.to_string()));
        };
        let tensor = unsafe { Tensor::from_tensor_info(ctx, tensor_info) }?;
        unsafe {
            (*tensor.0).backend = ggml_backend_type::GGML_BACKEND_GPU;
        }
        let nbytes = unsafe { ggml_nbytes(tensor.0) };
        let layout = std::alloc::Layout::array::<u8>(nbytes).unwrap();

        unsafe {
            (*tensor.0).data = std::alloc::alloc(layout) as *mut std::ffi::c_void;
        };

        let tensor_offset = tensor_info.inner.offset;

        gguf_file
            .seek(SeekFrom::Start(data_offset + tensor_offset))
            .unwrap();

        gguf_file
            .read_exact(unsafe {
                std::slice::from_raw_parts_mut(std::mem::transmute((*tensor.0).data), nbytes)
            })
            .unwrap();

        unsafe {
            ggml_cuda_transform_tensor((*tensor.0).data, tensor.0);
            std::alloc::dealloc(std::mem::transmute((*tensor.0).data), layout);
        }

        return Ok(Tensor(tensor.0, PhantomData));
    }

    fn get_key_value(&self, key: &str) -> Result<&GGUFKeyValue<'a>, Self::Error> {
        self.gguf_ctx
            .key_values
            .get(key)
            .ok_or(BuilderError::KeyValueNotFound(key.to_string()))
    }

    /*
    fn create_tensor_1d<const DIM: usize>(
        &mut self,
        name: String,
        dtype: ggml_type,
        dim: usize,
        tensor: &mut GTensor<'a>,
    ) -> Result<(), Self::Error> {
        if self.load_tensors.contains_key(name.as_str()) {
            return Err(BuilderError::DuplicateTensor);
        }

        self.create_tensors
            .insert(name, (1, [dim, 0, 0, 0], dtype, tensor));
        Ok(())
    }

    fn create_tensor_2d<const DIM: usize>(
        &mut self,
        name: String,
        dtype: ggml_type,
        dim: [usize; 2],
        tensor: &mut GTensor<'a>,
    ) -> Result<(), Self::Error> {
        if self.load_tensors.contains_key(name.as_str()) {
            return Err(BuilderError::DuplicateTensor);
        }

        self.create_tensors
            .insert(name, (2, [dim[0], dim[1], 0, 0], dtype, tensor));
        Ok(())
    }

    fn create_tensor_3d<const DIM: usize>(
        &mut self,
        name: String,
        dtype: ggml_type,
        dim: [usize; 3],
        tensor: &mut GTensor<'a>,
    ) -> Result<(), Self::Error> {
        if self.load_tensors.contains_key(name.as_str()) {
            return Err(BuilderError::DuplicateTensor);
        }

        self.create_tensors
            .insert(name, (3, [dim[0], dim[1], dim[2], 0], dtype, tensor));
        Ok(())
    }

    fn create_tensor_4d(
        &mut self,
        name: String,
        dtype: ggml_type,
        dim: [usize; 4],
        tensor: &mut GTensor<'a>,
    ) -> Result<(), Self::Error> {
        if self.load_tensors.contains_key(name.as_str()) {
            return Err(BuilderError::DuplicateTensor);
        }

        self.create_tensors.insert(name, (4, dim, dtype, tensor));
        Ok(())
    }*/

    /*
    fn load_tensor(
        &'a mut self,
        name: String,
        tensor: LCell<'id, GTensor<'a>>,
    ) -> Result<(), Self::Error> {
        if self.load_tensors.contains_key(name.as_str()) {
            return Err(BuilderError::DuplicateTensor);
        }

        self.load_tensors.insert(name, tensor);

        Ok(())
    }*/
}

/*
impl<'a> CudaBackendBuilder<'a> {
    pub fn build(self) -> Result<CudaBackend, GError> {
        let Self {
            backend,
            gguf_ctx,
            mut gguf_file,
        } = self
        else {
            panic!()
        };

        let tensors: Vec<_> = gguf_ctx.tensor_iter().filter(|t| with_tensors(t)).collect();

        let mem_size = tensors.len() * (GGML_OBJECT_SIZE + GGML_TENSOR_SIZE);
        let mem_buffer = std::ptr::null_mut();
        let no_alloc = true;

        let init_params = ggml_init_params {
            mem_size,
            mem_buffer,
            no_alloc,
        };

        let ctx = unsafe { ggml_init(init_params) };

        let data_offset = unsafe { gguf_get_data_offset(gguf_ctx.gguf_ctx) } as u64;

        for tensor_ in tensors {
            let tensor = unsafe { ggml_dup_tensor(ctx, tensor_.0) };
            unsafe {
                (*tensor).backend = ggml_backend_type_GGML_BACKEND_GPU;
            }
            unsafe {
                ggml_set_name(
                    tensor,
                    std::mem::transmute((*tensor_.0).name.as_slice().as_ptr()),
                );
            }
            let nbytes = unsafe { ggml_nbytes(tensor) };
            let layout = std::alloc::Layout::array::<u8>(nbytes).unwrap();
            let name = std::ffi::CStr::from_bytes_until_nul(unsafe {
                std::mem::transmute((*tensor).name.as_slice())
            })
            .unwrap();
            let name = name.to_str().unwrap();

            let tensor_info = gguf_ctx
                .tensor_info(name)
                .ok_or(GError::GGUFTensorNotFound(name.into()))?;

            unsafe {
                (*tensor).data = std::alloc::alloc(layout) as *mut std::ffi::c_void;
            };

            let tensor_offset = tensor_info.inner.offset;

            gguf_file
                .seek(SeekFrom::Start(data_offset + tensor_offset))
                .unwrap();

            gguf_file
                .read_exact(unsafe {
                    std::slice::from_raw_parts_mut(std::mem::transmute((*tensor).data), nbytes)
                })
                .unwrap();

            unsafe {
                ggml_cuda_transform_tensor((*tensor).data, tensor);
                std::alloc::dealloc(std::mem::transmute((*tensor).data), layout);
            }
        }

        Ok(CudaBackend {
            ggml_backend: backend,
            ggml_context: ctx,
        })
    }
}

impl CudaBackend {
    pub fn builder(
        device: CudaDevice,
    ) -> Result<CudaBackendBuilder<'static, false, false>, GError> {
    }

    pub fn devices() -> impl Iterator<Item = CudaDevice> {
        CudaDeviceIter::new()
    }
}

impl std::ops::Drop for CudaBackend {
    fn drop(&mut self) {
        unsafe {
            for tensor in TensorsIterator::new(self.ggml_context) {
                ggml_cuda_free_data(tensor.0);
            }
        }
    }
}*/
