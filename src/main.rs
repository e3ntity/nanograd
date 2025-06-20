mod plot;
mod scalar;

use crate::plot::dump_graph;
use scalar::Scalar;
use scalar::func;

use rand::Rng;

fn rand_normal() -> f64 {
    let mut rng = rand::rng();

    let u1: f64 = rng.random_range(0.0..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

struct Perceptron<const N: usize> {
    bias: Scalar,
    weights: [Scalar; N],
}

impl<const N: usize> Perceptron<N> {
    fn new() -> Self {
        let weights = std::array::from_fn(|_| Scalar::new_grad(rand_normal() as f32));

        Perceptron {
            bias: Scalar::new_grad(rand_normal() as f32),
            weights,
        }
    }

    fn forward(&self, val: [Scalar; N]) -> Scalar {
        let mut out = Scalar::new(0.);
        for i in 0..N {
            out = out + self.weights[i].clone() * val[i].clone();
        }
        out = out + self.bias.clone();

        out
    }

    fn params(&self) -> Vec<Scalar> {
        let mut params: Vec<Scalar> = self.weights.clone().into_iter().collect();
        params.push(self.bias.clone());

        params
    }
}

fn forward<const N: usize, const M: usize>(
    layer: &[Perceptron<N>; M],
    input: [Scalar; N],
) -> [Scalar; M] {
    std::array::from_fn(|i| layer[i].forward(input.clone()))
}

fn main() {
    // Create a perceptron and dump its computational graph to a file.
    let perceptron = Perceptron::new();
    let z = perceptron.forward([Scalar::new(3.0), Scalar::new(2.0)]);
    z.backward();

    if let Err(e) = dump_graph(&z, "graph.html") {
        eprintln!("Failed to dump graph: {e}");
    };

    // Learn the XOR function with a super simple perceptron
    // (Training does *not* converge every time)
    let l1: [Perceptron<2>; 4] = std::array::from_fn(|_| Perceptron::new());
    let l2: [Perceptron<4>; 1] = std::array::from_fn(|_| Perceptron::new());

    let mut params: Vec<Scalar> = Vec::new();
    for p in l1.iter() {
        params.extend(p.params());
    }
    for p in l2.iter() {
        params.extend(p.params());
    }

    let fw = |x| {
        let h1_out = forward(&l1, x);
        let h1_act = std::array::from_fn(|i| func::relu(h1_out[i].clone()));

        let h2_out = forward(&l2, h1_act);

        func::sigmoid(h2_out[0].clone())
    };
    let step = |lr: f32| {
        for p in params.iter() {
            p.set_value(p.get_value() - lr * p.get_grad().unwrap());
        }
    };
    let zero_grad = || {
        for p in params.iter() {
            p.set_grad(0.);
        }
    };

    let training_data = vec![[0., 1., 1.], [0., 0., 0.], [1., 0., 1.], [1., 1., 0.]];

    for epoch in 0..1000 {
        let mut epoch_loss = Scalar::new(0.);
        for data in training_data.clone() {
            let input = [Scalar::new(data[0]), Scalar::new(data[1])];
            let label = Scalar::new(data[2]);

            let prediction = fw(input);
            let one = Scalar::new(1.0);
            let loss = -(label.clone() * func::ln(prediction.clone())
                + (one.clone() - label.clone()) * func::ln(one - prediction.clone()));

            epoch_loss = epoch_loss + loss;
        }
        epoch_loss = epoch_loss * (1. / training_data.len() as f32);

        if epoch % 100 == 0 {
            eprintln!("Epoch {}: loss={}", epoch, epoch_loss.get_value());
        }

        zero_grad();
        epoch_loss.backward();
        step(0.2);
    }

    for dp in training_data.iter() {
        let input = [Scalar::new(dp[0]), Scalar::new(dp[1])];
        let out = fw(input);

        eprintln!("{} XOR {} = {}", dp[0], dp[1], out.get_value());
    }
}
