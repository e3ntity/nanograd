use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::perceptron::Perceptron;
use crate::scalar::Scalar;
use crate::scalar::func;
use crate::util::plot::dump_graph;

pub fn run() {
    // Create a perceptron and dump its computational graph to a file.
    let perceptron = Perceptron::new();
    let z = perceptron.forward([Scalar::new(3.0), Scalar::new(2.0)]);
    z.backward();

    if let Err(e) = dump_graph(&(z), "perceptron_graph.html") {
        eprintln!("Failed to dump graph: {e}");
    };

    // Learn the XOR function with a super simple perceptron
    // (Training does *not* converge every time)
    let l1: Linear<2, 4> = Linear::new();
    let l2: Linear<4, 1> = Linear::new();

    let mut params: Vec<Scalar> = Vec::new();
    params.extend(l1.params());
    params.extend(l2.params());

    let fw = |x| {
        let h1_out = l1.forward(x);
        let h1_act = std::array::from_fn(|i| func::relu(h1_out[i].clone()));

        let h2_out = l2.forward(h1_act);

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
