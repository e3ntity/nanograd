pub mod dataset;
pub mod mnist;
pub mod network;
pub mod server;

pub use mnist::{serve, train};
