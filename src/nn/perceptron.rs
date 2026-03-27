use super::module::Module;
use crate::scalar::Scalar;
use crate::util::dist::rand_normal;

#[derive(Clone, Debug)]
pub struct Perceptron<const N: usize> {
    bias: Scalar,
    weights: [Scalar; N],
}

impl<const N: usize> Perceptron<N> {
    pub fn new() -> Self {
        let bias = Scalar::new_grad(rand_normal() as f32);
        let weights = std::array::from_fn(|_| Scalar::new_grad(rand_normal() as f32));

        Perceptron { bias, weights }
    }

    pub fn forward(&self, val: [Scalar; N]) -> Scalar {
        let mut out = self.bias.clone();
        for i in 0..N {
            out = out + self.weights[i].clone() * val[i].clone();
        }

        out
    }
}

impl<const N: usize> Module for Perceptron<N> {
    fn params(&self) -> Vec<Scalar> {
        let mut params: Vec<Scalar> = self.weights.clone().into_iter().collect();
        params.push(self.bias.clone());

        params
    }
}
