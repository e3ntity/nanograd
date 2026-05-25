use super::module::Module;
use super::perceptron::Perceptron;
use crate::scalar::Scalar;
use crate::util::dist::rand_normal;

#[derive(Clone, Debug)]
pub struct Linear<const I: usize, const O: usize> {
    perceptrons: [Perceptron<I>; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    pub fn new() -> Self {
        let perceptrons = std::array::from_fn(|_| Perceptron::new());
        let ffw = Linear { perceptrons };

        for param in ffw.params() {
            param.set_value(rand_normal() as f32 * 1.0 / (I as f32).sqrt());
        }

        ffw
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
