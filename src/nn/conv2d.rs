use crate::{
    nn::{
        filter2d::{self, Filter2d},
        module::Module,
    },
    scalar::Scalar,
    util::dist::rand_normal,
};

#[derive(Clone, Debug)]
pub struct Conv2d<const CI: usize, const CO: usize, const K: usize> {
    filters: [Filter2d<CI, K>; CO],
}

impl<const CI: usize, const CO: usize, const K: usize> Conv2d<CI, CO, K> {
    pub fn new() -> Conv2d<CI, CO, K> {
        let filters = std::array::from_fn(|_| Filter2d::new());

        let conv = Conv2d { filters };
        for param in conv.params() {
            param.set_value(rand_normal() as f32 * 1.0 / ((CI * K * K) as f32).sqrt());
        }

        conv
    }

    pub fn forward(&self, x: [Vec<Vec<Scalar>>; CI]) -> [Vec<Vec<Scalar>>; CO] {
        std::array::from_fn(|i| self.filters[i].forward(x.clone()))
    }
}

impl<const CI: usize, const CO: usize, const K: usize> Module for Conv2d<CI, CO, K> {
    fn params(&self) -> Vec<Scalar> {
        let mut params: Vec<Scalar> = vec![];
        for filter in &self.filters {
            params.extend(filter.params());
        }

        params
    }
}

pub fn max_pool_2d<const C: usize>(
    x: [Vec<Vec<Scalar>>; C],
    kernel: usize,
    stride: usize,
) -> [Vec<Vec<Scalar>>; C] {
    std::array::from_fn(|i| filter2d::max_pool_2d(x[i].clone(), kernel, stride))
}
