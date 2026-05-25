use std::fs::File;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use rand::rng;
use rand::seq::SliceRandom;

use crate::examples::mnist::dataset::{DatasetSplit, load_image, load_mnist_images};
use crate::examples::mnist::network::Network;
use crate::examples::mnist::server::{self, HttpMethod, Request};
use crate::scalar::{Scalar, func};

fn loss_l2<const N: usize>(pred: [Scalar; N], target: [Scalar; N]) -> Scalar {
    std::array::from_fn::<_, N, _>(|i| {
        (target[i].clone() - pred[i].clone()) * (target[i].clone() - pred[i].clone())
    })
    .iter()
    .fold(Scalar::new(0.0), |agg, v| agg + (*v).clone())
        * Scalar::new(1.0 / target.len() as f32)
}

fn loss_ce<const N: usize>(pred: [Scalar; N], label: usize) -> Scalar {
    -func::ln(pred[label].clone())
}

pub fn train(path: String) {
    let epochs: usize = 100;
    let batch_size: usize = 32;
    let learning_rate: f32 = 1e-3;
    let weight_decay: f32 = 1e-4;
    let momentum = 0.9;
    let test_count = 256;
    let batch_count = 100;

    let mut model = Network::new(Some(momentum));

    eprintln!("Model: {:.1}K params", model.params().len() as f32 / 1e3);
    eprintln!(
        "Training: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}"
    );

    let mut data_train = load_mnist_images(DatasetSplit::Train).expect("Load train dataset");
    let mut data_test = load_mnist_images(DatasetSplit::Test).expect("Load test dataset");
    data_test.shuffle(&mut rng());

    let mut best_score = 0.0;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        data_train.shuffle(&mut rng());
        for batch in 0..batch_count {
            let mut loss = Scalar::new(0.0);
            for step in 0..batch_size {
                let (image, label) = data_train[batch * batch_size + step].clone();
                let pred = model.forward(image);
                let loss_weight_decay = model
                    .params()
                    .iter()
                    .fold(Scalar::new(0.0), |agg, p| agg + p.clone() * p.clone())
                    * Scalar::new(1.0 / model.params().len() as f32);

                loss = loss + loss_ce(pred, label as usize) + weight_decay * loss_weight_decay;
            }

            let completion = ((batch + 1) as f32) / (batch_count as f32);
            let completed = (30.0 * completion) as usize;
            eprint!(
                "Epoch {}: train [{}{}] {}%\r",
                epoch,
                str::repeat("=", completed),
                str::repeat(" ", 30 - completed),
                (completion * 100.0) as usize
            );
            loss = loss * (1.0 / batch_size as f32);

            model.grad_zero();
            loss.backward();
            model.grad_step(learning_rate);

            total_loss += loss.get_value();
        }

        let mut correct = 0;
        for (step, (image, label)) in data_test[..test_count].iter().enumerate() {
            let (pred, _) = model
                .forward(image.clone())
                .iter()
                .enumerate()
                .max_by(|(_, v1), (_, v2)| v1.get_value().total_cmp(&v2.get_value()))
                .unwrap();

            if pred == (*label as usize) {
                correct += 1;
            }

            let completion = ((step + 1) as f32) / (test_count as f32);
            let completed = (30.0 * completion) as usize;
            eprint!(
                "Epoch {}: test [{}{}] {}%\r",
                epoch,
                str::repeat("=", completed),
                str::repeat(" ", 30 - completed),
                (completion * 100.0) as usize
            );
        }
        let score = (correct as f32) / (test_count as f32);

        eprintln!(
            "Epoch {}: loss={:.4}, score={:.4}",
            epoch,
            total_loss / (batch_count as f32),
            score,
        );

        if score <= best_score {
            continue;
        }

        best_score = score;
        eprintln!("New best, saving to {path}..");
        model.save(path.clone()).unwrap();
    }

    println!("Training finished, saved model with min_loss={best_score} to \"{path}\"");
}

pub fn serve(path: String) {
    let host = "0.0.0.0";
    let port = 8200;

    let mut model = Network::new(None);
    model
        .load(path.clone())
        .expect(&*format!("Failed to load model from {path}"));

    let listener = TcpListener::bind(format!("{host}:{port}"))
        .expect(&*format!("Failed to listen on {host}:{port}"));
    println!("Serving \"{path}\" on {host}:{port}");

    for stream in listener.incoming() {
        match stream {
            Ok(conn) => server::handle_connection(conn, model.clone()),
            Err(e) => println!("Connection failed: {e}"),
        }
    }
}
