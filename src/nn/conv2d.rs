use crate::{
    nn::{filter2d::Filter2d, module::Module},
    scalar::Scalar,
};

#[derive(Clone, Debug)]
pub struct Conv2d<const CI: usize, const CO: usize, const K: usize> {
    filters: [Filter2d<CI, K>; CO],
}

impl<const CI: usize, const CO: usize, const K: usize> Conv2d<CI, CO, K> {
    pub fn new() -> Conv2d<CI, CO, K> {
        let filters = std::array::from_fn(|_| Filter2d::new());

        Conv2d { filters }
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
