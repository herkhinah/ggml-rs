use crate::{
    gguf::{GGUFError, GGUFKeyValue, GGUFTensorInfo},
    tensor::Tensor,
};

pub trait Builder<'a> {
    type Error: std::fmt::Debug + From<GGUFError>;

    //fn tensors(&self) -> TensorsIterator<'_>;
    fn get_tensor_info(&self, name: &str) -> Result<&GGUFTensorInfo<'a>, Self::Error>;
    //fn get_tensor(&self, name: &str) -> Result<&GTensor<'a>, Self::Error>;
    fn get_key_value(&self, key: &str) -> Result<&GGUFKeyValue<'a>, Self::Error>;

    fn alloc(&mut self, tensors: usize);

    fn load_tensor(&mut self, name: &str) -> Result<Tensor<'a>, Self::Error>;
}
