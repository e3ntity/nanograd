use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use crate::nn::conv2d::{Conv2d, max_pool_2d};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::scalar::{Scalar, func};
use crate::util::dist::rand_normal;

fn elementwise_t1<const C: usize, T, U>(x: [Vec<T>; C], f: impl Fn(T) -> U) -> [Vec<U>; C] {
    x.map(|ch| ch.into_iter().map(&f).collect())
}

fn elementwise_t2<const C: usize, T, U>(
    x: [Vec<Vec<T>>; C],
    f: impl Fn(T) -> U,
) -> [Vec<Vec<U>>; C] {
    x.map(|ch| {
        ch.into_iter()
            .map(|row| row.into_iter().map(&f).collect())
            .collect()
    })
}

#[derive(Clone)]
pub struct Network {
    conv1: Conv2d<1, 8, 3>,
    conv2: Conv2d<8, 16, 2>,
    fc1: Linear<576, 10>,
    momentum_coeff: f32,
    velocity: Option<Vec<f32>>,
}

impl Network {
    pub fn new(momentum_coeff: Option<f32>) -> Network {
        let conv1 = Conv2d::new();
        let conv2 = Conv2d::new();
        let fc1 = Linear::new();

        Network {
            conv1,
            conv2,
            fc1,
            momentum_coeff: momentum_coeff.unwrap_or(0.9),
            velocity: None,
        }
    }

    pub fn forward(&self, x: Vec<Vec<Scalar>>) -> [Scalar; 10] {
        let out_conv1 = elementwise_t2(self.conv1.forward([x]), |v| func::relu(v));
        let out_mp1 = max_pool_2d(out_conv1, 2, 2);
        let out_conv2 = elementwise_t2(self.conv2.forward(out_mp1), |v| func::relu(v));
        let out_mp2 = max_pool_2d(out_conv2, 2, 2);
        let out_conv = out_mp2;

        assert_eq!(
            out_conv.len() * out_conv[0].len() * out_conv[0][0].len(),
            576
        );

        let embedding: [Scalar; 576] = std::array::from_fn(|idx| {
            let i = idx / (out_conv[0].len() * out_conv[0][0].len());
            let j = (idx / out_conv[0][0].len()) % out_conv[0].len();
            let k = idx % out_conv[0][0].len();

            out_conv[i][j][k].clone()
        });

        let out_fc1 = self.fc1.forward(embedding);
        let out = func::softmax(out_fc1);

        out
    }

    pub fn params(&self) -> Vec<Scalar> {
        self.conv1
            .params()
            .iter()
            .cloned()
            .chain(self.conv2.params().iter().cloned())
            .chain(self.fc1.params().iter().cloned())
            .collect()
    }

    fn get_velocity(&self) -> Vec<f32> {
        match self.velocity.clone() {
            Some(v) => v,
            None => self.params().iter().map(|_| 0.0 as f32).collect(),
        }
    }

    pub fn grad_step(&mut self, lr: f32) {
        let mut velocity = self.get_velocity();

        let params = self.params();
        for i in 0..params.len() {
            velocity[i] = self.momentum_coeff * velocity[i] + params[i].get_grad().unwrap();
            params[i].set_value(params[i].get_value() - lr * velocity[i]);
        }

        self.velocity = Some(velocity);
    }

    pub fn grad_zero(&self) {
        for p in self.params().iter() {
            p.set_grad(0.);
        }
    }

    pub fn save(&self, path: String) -> std::io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        for v in self.get_velocity() {
            writer.write_all(&v.to_le_bytes())?;
        }

        for x in self.params().iter() {
            writer.write_all(&x.get_value().to_le_bytes())?;
        }

        Ok(())
    }

    pub fn load(&mut self, path: String) -> std::io::Result<()> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;

        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let mut velocity = self.get_velocity();
        for i in 0..velocity.len() {
            velocity[i] = data[i];
        }
        self.velocity = Some(velocity.clone());

        for (i, p) in self.params().iter().enumerate() {
            p.set_value(data[velocity.len() + i]);
        }

        Ok(())
    }
}
