#![allow(dead_code, unused_imports, unused_variables, unused_mut, unused_assignments)]

/* Module imports */
mod network;
pub mod activation;
pub mod mnist_reader;
pub mod utils;

/* Imports */
use crate::mnist_reader::{ Mnist, format_data };

fn main() {
    /* Test data */
    let mnist_data = Mnist::new("dataset/");
    let data: Vec<(Vec<f64>, Vec<f64>)> = format_data(&mnist_data.test_data, &mnist_data.test_labels);

    /* Neural net */
    let mut network = network::Network::new_kaiming_init(vec![784, 16, 16, 10], 0.000000005, 0.9, 0.1);
    network.train(data);
}
