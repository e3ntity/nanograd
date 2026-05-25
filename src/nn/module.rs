use crate::scalar::Scalar;

pub trait Module {
    fn params(&self) -> Vec<Scalar>;
}
