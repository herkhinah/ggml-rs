use crate::{gguf::GGUFTensorInfo, tensor::GTensor, TensorsIterator};

pub trait Builder<'a> {
    type Error: std::fmt::Debug;

    fn tensors(&self) -> TensorsIterator<'_>;
    fn tensor_info(&self, name: &str) -> Option<&GGUFTensorInfo<'a>>;

    fn alloc(&mut self, tensors: usize);

    fn load_tensor(&mut self, name: &str) -> Result<GTensor<'a>, Self::Error>;
}