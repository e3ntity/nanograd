use crate::tensor::error::TensorError;

#[derive(Debug)]
pub enum Error {
    Tensor(TensorError),
}

impl From<TensorError> for Error {
    fn from(value: TensorError) -> Self {
        Error::Tensor(value)
    }
}
