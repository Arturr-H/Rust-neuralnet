#![allow(dead_code, unused_imports, unused_variables)]

/* Imports */
pub mod utils;
mod mnist_reader;
use activation::{Activation, NetworkActivations};
use cost::Cost;
use learn_rate::LearnRate;
use mnist_reader::{ Mnist, format_data };
use crate::{mnist_reader::print_image, network::Network};
use serde_derive::{ Serialize, Deserialize };

#[path = "./network/network.rs"]
mod network;

#[path = "./network/layer.rs"]
mod layer;

#[path = "./network/activation.rs"]
mod activation;

#[path = "./network/cost.rs"]
mod cost;

#[path = "./network/learn_data.rs"]
mod learn_data;

#[path = "./network/learn_rate.rs"]
mod learn_rate;

#[path = "./datavis/handler.rs"]
mod handler;

fn main() -> () {
    let mut nn = network::Network::new(
        [784, 128, 128, 10],
        100,
        LearnRate::new(0.05, 0.001),
        0.9,
        0.1,
        cost::CostType::CrossEntropy,
        NetworkActivations::new(Activation::leaky_relu(), Activation::sigmoid())
    );
    let mnist_data = Mnist::new("dataset/");
    // let data: Vec<(Vec<f64>, Vec<f64>)> = format_data(&mnist_data.train_data, &mnist_data.train_labels);
    // nn.train(data);
    let mut nn = Network::retrieve_from_save("./src/saves/network");


    let test_data: Vec<(Vec<f64>, Vec<f64>)> = format_data(&mnist_data.train_data, &mnist_data.train_labels);
    for (input, expected_output) in test_data {
        let prediction = nn.classify(&input);
        let net_predict = prediction.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index).unwrap();

        let expected = expected_output.iter().position(|&e| e == 1.0).unwrap() as u8;
        print_image(input, expected);
        println!("Network {net_predict}");
        println!("{}", if net_predict == expected as usize { "✅" } else { "🚫" });
        
        let mut a = String::new();
        let _ = match std::io::stdin().read_line(&mut a) {
            Ok(_) => {
                continue
            },
            Err(_) => continue,
        };
    }
}


