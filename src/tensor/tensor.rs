use std::ops;

use crate::{scalar::Scalar, tensor::error::TensorError};

type Result<T> = core::result::Result<T, TensorError>;

#[derive(Clone, Debug)]
pub struct Tensor<const D: usize> {
    data: Vec<Scalar>,
    size: [usize; D],
}

impl<const D: usize> Tensor<D> {
    pub fn zeros(size: [usize; D]) -> Tensor<D> {
        let elements = size.iter().product();
        let data = (0..elements).map(|_| Scalar::new(0.0)).collect();

        Tensor { data, size }
    }

    pub fn enable_grad(&self) -> () {
        self.data.iter().for_each(|s| s.set_grad(0.0));
    }

    /// Subtracts the right-hand side vector from the vector and returns the result.
    pub fn try_sub(&self, rhs: &Tensor<D>) -> Result<Tensor<D>> {
        self.try_add(&(-rhs))
    }

    /// Adds vectors element-wise and returns the result.
    pub fn try_add(&self, rhs: &Tensor<D>) -> Result<Tensor<D>> {
        if self.size != rhs.size {
            return Err(TensorError::ShapeMismatch {
                expected: self.size.clone().into(),
                got: rhs.size.clone().into(),
            });
        }

        let data: Vec<Scalar> = (0..self.data.len())
            .map(|i| self.data[i].clone() + rhs.data[i].clone())
            .collect();
        let size = self.size;

        Ok(Tensor { data, size })
    }

    /// Multiplies vectors element-wise and returns the result.
    pub fn try_mul(&self, rhs: &Tensor<D>) -> Result<Tensor<D>> {
        if self.size != rhs.size {
            return Err(TensorError::ShapeMismatch {
                expected: self.size.clone().into(),
                got: rhs.size.clone().into(),
            });
        }

        let data: Vec<Scalar> = (0..self.data.len())
            .map(|i| self.data[i].clone() * rhs.data[i].clone())
            .collect();
        let size = self.size;

        Ok(Tensor { data, size })
    }
}

impl<const D: usize> ops::Add<&Scalar> for &Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: &Scalar) -> Self::Output {
        let data = self.data.iter().cloned().map(|v| v + rhs.clone()).collect();
        let size = self.size.clone();

        Tensor { data, size }
    }
}

impl<const D: usize> ops::Neg for &Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        let data = self.data.iter().cloned().map(|v| -v).collect();
        let size = self.size.clone();

        Tensor { data, size }
    }
}
