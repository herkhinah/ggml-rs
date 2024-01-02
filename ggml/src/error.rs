use std::ffi::FromBytesUntilNulError;

use thiserror::Error;

use crate::gguf::GGUFError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("expected cpu backend")]
    BackendNotCpu,
    #[error("failed to initialize the cublas backend")]
    CublasInitFailed,
    #[error("cublas wasn't loaded")]
    CublasNotLoaded,
    #[error("failed to initialize the cpu backend")]
    CpuBackendInitFailed,
    #[error("couldn't find tensor {0} in GGUF file")]
    GGUFTensorNotFound(String),
    #[error("invalid tensor name")]
    InvalidTensorName(FromBytesUntilNulError),
    #[error("failed to open file")]
    FileError(std::io::Error),
    #[error("gguf error")]
    GGUFError(#[from] GGUFError),
    #[error("ggml_new_tensor failed")]
    GGMLNewTensorFailed,
}
