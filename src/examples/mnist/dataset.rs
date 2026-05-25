use rand::Rng;
use std::{fs::DirEntry, io};

use crate::scalar::Scalar;

#[derive(Debug)]
pub enum MnistError {
    Image(image::ImageError),
    InvalidInput(String),
    Io(io::Error),
}

impl From<image::ImageError> for MnistError {
    fn from(e: image::ImageError) -> Self {
        Self::Image(e)
    }
}

impl From<io::Error> for MnistError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

pub enum DatasetSplit {
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

impl std::fmt::Display for MnistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image(e) => write!(f, "Image error: {e}"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

pub fn load_image(path: String) -> Result<Vec<Vec<Scalar>>, MnistError> {
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

pub fn load_mnist_images(split: DatasetSplit) -> Result<Vec<(Vec<Vec<Scalar>>, u8)>, MnistError> {
    let mut data: Vec<(Vec<Vec<Scalar>>, u8)> = vec![];

    for label in 0..10 {
        let directory = format!("data/mnist/{}/{label}", split.to_string());
        let entries: Vec<DirEntry> = std::fs::read_dir(directory)
            .map_err(MnistError::Io)?
            .filter_map(Result::ok)
            .collect();

        data.reserve(entries.len());
        for entry in entries {
            let path = entry.path();
            let path_str = path
                .to_str()
                .ok_or_else(|| MnistError::InvalidInput(format!("Invalid path: {path:?}")))?
                .to_owned();

            data.push((load_image(path_str)?, label));
        }
    }

    Ok(data)
}
