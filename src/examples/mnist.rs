use rand::Rng;
use std::fs::DirEntry;
use std::io;

use crate::nn::conv2d::Conv2d;
use crate::nn::filter2d::Filter2d;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::scalar::{Scalar, func};
use crate::util::plot::dump_graph;

#[derive(Debug)]
enum DatasetError {
    Image(image::ImageError),
    InvalidInput(String),
    Io(io::Error),
}

impl From<image::ImageError> for DatasetError {
    fn from(e: image::ImageError) -> Self {
        Self::Image(e)
    }
}

impl From<io::Error> for DatasetError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

enum DatasetSplit {
    Test,
    Train,
}

impl std::fmt::Display for DatasetSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            DatasetSplit::Test => "test",
            DatasetSplit::Train => "train",
        }
        .to_owned();

        write!(f, "{v}")
    }
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image(e) => write!(f, "Image error: {e}"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

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

fn load_image(path: String) -> Result<Vec<Vec<Scalar>>, DatasetError> {
    let img = image::open(path)?.to_luma8();
    let (h, w) = img.dimensions();

    let mut out: Vec<Vec<Scalar>> = (0..h)
        .map(|_| (0..w).map(|_| Scalar::new(0.0)).collect())
        .collect();
    for px in 0..h * w {
        let y = px / w;
        let x = px % w;
        let pixel = img[(x, y)].0[0];
        out[y as usize][x as usize] = Scalar::new((pixel as f32) / 255.0);
    }

    Result::Ok(out)
}

fn sample_mnist_image(
    split: DatasetSplit,
) -> Result<(String, (Vec<Vec<Scalar>>, u8)), DatasetError> {
    let label = rand::rng().random_range(0..10);
    let directory = format!("data/mnist/{}/{label}", split.to_string());

    let entries: Vec<DirEntry> = std::fs::read_dir(directory)
        .map_err(DatasetError::Io)?
        .filter_map(Result::ok)
        .collect();
    let entry = &entries[rand::rng().random_range(0..entries.len())];

    let path = entry.path();
    let path_str = path
        .to_str()
        .ok_or_else(|| DatasetError::InvalidInput(format!("Invalid path: {path:?}")))?
        .to_owned();
    let image = load_image(path_str.clone())?;

    Ok((path_str, (image, label)))
}

fn one_hot<const N: usize>(label: usize) -> [Scalar; N] {
    std::array::from_fn(|i| Scalar::new(if i == label { 1.0 } else { 0.0 }))
}

struct Network {
    conv1: Conv2d<1, 4, 6>,
    conv2: Conv2d<4, 8, 6>,
    conv3: Conv2d<8, 16, 6>,
    conv4: Conv2d<16, 32, 13>,
    fc1: Linear<32, 10>,
}

impl Network {
    pub fn new() -> Network {
        let conv1 = Conv2d::new();
        let conv2 = Conv2d::new();
        let conv3 = Conv2d::new();
        let conv4 = Conv2d::new();
        let fc1 = Linear::new();

        Network {
            conv1,
            conv2,
            conv3,
            conv4,
            fc1,
        }
    }

    pub fn forward(&self, x: Vec<Vec<Scalar>>) -> [Scalar; 10] {
        let out_conv1 = elementwise_t2(self.conv1.forward([x]), |v| func::sigmoid(v));
        let out_conv2 = elementwise_t2(self.conv2.forward(out_conv1), |v| func::sigmoid(v));
        let out_conv3 = elementwise_t2(self.conv3.forward(out_conv2), |v| func::sigmoid(v));
        let out_conv4 = elementwise_t2(self.conv4.forward(out_conv3), |v| func::sigmoid(v));

        let (l1, l2) = (out_conv4[0].len(), out_conv4[0][0].len());
        if l1 != 1 || l2 != 1 {
            panic!("Invalid conv output size {}x{l1}x{l2}", out_conv4.len());
        }

        let out_conv = out_conv4.map(|v| v[0][0].clone());

        let out_fc1 = self.fc1.forward(out_conv);
        let out = func::softmax(out_fc1);

        out
    }

    pub fn params(&self) -> Vec<Scalar> {
        self.conv1
            .params()
            .iter()
            .cloned()
            .chain(self.conv2.params().iter().cloned())
            .chain(self.conv3.params().iter().cloned())
            .chain(self.conv4.params().iter().cloned())
            .chain(self.fc1.params().iter().cloned())
            .collect()
    }

    pub fn grad_step(&self, lr: f32) {
        for p in self.params().iter() {
            p.set_value(p.get_value() - lr * p.get_grad().unwrap());
        }
    }

    pub fn grad_zero(&self) {
        for p in self.params().iter() {
            p.set_grad(0.);
        }
    }
}

fn loss_l2<const N: usize>(pred: [Scalar; N], target: [Scalar; N]) -> Scalar {
    std::array::from_fn::<_, N, _>(|i| {
        (target[i].clone() - pred[i].clone()) * (target[i].clone() - pred[i].clone())
    })
    .iter()
    .fold(Scalar::new(0.0), |agg, v| agg + (*v).clone())
        * Scalar::new(1.0 / target.len() as f32)
}

fn loss_ce<const N: usize>(pred: [Scalar; N], target: [Scalar; N]) -> Scalar {
    -std::array::from_fn::<_, N, _>(|i| target[i].clone() * func::ln(pred[i].clone()))
        .iter()
        .fold(Scalar::new(0.0), |agg, v| agg + (*v).clone())
}

pub fn run() {
    let epochs: usize = 20;
    let batch_size: usize = 32;
    let learning_rate: f32 = 0.001;
    let weight_decay: f32 = 0.1;
    let model = Network::new();

    eprintln!("Model: {:.1}K params", model.params().len() as f32 / 1e3);
    eprintln!(
        "Training: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}"
    );

    let Ok((_, (sample_image, _))) = sample_mnist_image(DatasetSplit::Train) else {
        eprintln!("Failed to open mnist sample image");
        return;
    };
    let z = model
        .forward(sample_image)
        .iter()
        .fold(Scalar::new(0.0), |agg, v| agg + (*v).clone());

    if let Err(e) = dump_graph(&z, "mnist_model.html") {
        eprintln!("Failed to dump graph: {e}");
    };

    for epoch in 0..epochs {
        let mut loss = Scalar::new(0.0);

        for step in 0..batch_size {
            let Ok((_, (image, label_))) = sample_mnist_image(DatasetSplit::Train) else {
                eprintln!("Failed to open image");
                break;
            };
            let target: [Scalar; 10] = one_hot(label_ as usize);

            let pred = model.forward(image);
            let loss_weight_decay = model
                .params()
                .iter()
                .fold(Scalar::new(0.0), |agg, p| agg + p.clone() * p.clone())
                * Scalar::new(1.0 / model.params().len() as f32);

            loss = loss + loss_ce(pred, target) + weight_decay * loss_weight_decay;

            eprint!(
                "Epoch {}: [{}{}]\r",
                epoch,
                str::repeat("=", step + 1),
                str::repeat(" ", batch_size - step - 1)
            );
        }
        loss = loss * (1.0 / batch_size as f32);

        eprintln!("Epoch {}: loss={}", epoch, loss.get_value());

        model.grad_zero();
        loss.backward();
        model.grad_step(learning_rate);
    }
}
