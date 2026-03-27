#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    IndexOutOfBounds {
        index: usize,
        size: usize,
    },
}
