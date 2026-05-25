use super::module::Module;
use super::perceptron::Perceptron;
use crate::scalar::Scalar;
use crate::util::dist::rand_normal;

#[derive(Clone, Debug)]
pub struct Filter2d<const C: usize, const K: usize> {
    bias: Scalar,
    weights: [[[Scalar; K]; K]; C],
}

impl<const C: usize, const K: usize> Filter2d<C, K> {
    pub fn new() -> Self {
        let bias = Scalar::new_grad(rand_normal() as f32);
        let weights = std::array::from_fn(|_| {
            std::array::from_fn(|_| std::array::from_fn(|_| Scalar::new_grad(rand_normal() as f32)))
        });

        Filter2d { bias, weights }
    }

    pub fn forward(&self, x: [Vec<Vec<Scalar>>; C]) -> Vec<Vec<Scalar>> {
        let h_out = x[0].len() - K + 1;
        let w_out = x[0][0].len() - K + 1;

        let mut out: Vec<Vec<Scalar>> = vec![];
        out.reserve(h_out);

        for h in 0..h_out {
            out.push(vec![]);
            out[h].reserve(w_out);

            for w in 0..w_out {
                let mut acc = self.bias.clone();
                for h_ in 0..K {
                    for w_ in 0..K {
                        for c_ in 0..C {
                            acc = acc
                                + self.weights[c_][h_][w_].clone() * x[c_][h + h_][w + w_].clone();
                        }
                    }
                }

                out[h].push(acc);
            }
        }

        out
    }
}

impl<const C: usize, const K: usize> Module for Filter2d<C, K> {
    fn params(&self) -> Vec<Scalar> {
        let mut params: Vec<Scalar> = vec![self.bias.clone()];
        params.extend(self.weights.iter().flatten().flatten().cloned());

        params
    }
}
