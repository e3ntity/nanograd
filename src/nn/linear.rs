use super::module::Module;
use super::perceptron::Perceptron;
use crate::scalar::Scalar;

#[derive(Clone, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    perceptrons: [Perceptron<I>; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    pub fn new() -> Self {
        let perceptrons = std::array::from_fn(|_| Perceptron::new());
        Linear { perceptrons }
    }

    pub fn forward(&self, x: [Scalar; I]) -> [Scalar; O] {
        std::array::from_fn(|i| self.perceptrons[i].forward(x.clone()))
    }
}

impl<const I: usize, const O: usize> Module for Linear<I, O> {
    fn params(&self) -> Vec<Scalar> {
        self.perceptrons
            .iter()
            .map(|p| p.params())
            .flatten()
            .collect()
    }
}
