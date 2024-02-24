#![allow(dead_code, unused_imports, unused_variables)]

/* Imports */
pub mod utils;
mod data_handler;

mod mnist_reader;
use activation::{Activation, ActivationType, NetworkActivations};
use cost::Cost;
use data_handler::{load_data, modify_images, print_matrix, save_data, save_weight_layer_image};
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
        Some("./src/saves/network_v2"),
        [100, 100],
        40,
        LearnRate::new_range_with_step(0.1..0.000001, 2.0),
        0.9,
        0.1,
        cost::CostType::CrossEntropy,
        NetworkActivations::new(ActivationType::LeakyRelu, ActivationType::Softmax),
        is_correct
    );
    let test_data = load_data("./dataset/transformed/test_data");
    let train_data = load_data("./dataset/transformed/train_data");
    nn.train(train_data);

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
fn is_correct(input: &Vec<f64>, expected: &Vec<f64>) -> bool {
    let predict_max_idx = input
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index).unwrap();
    let expect_idx = expected.iter().position(|&e| e == 1.0).unwrap();

    predict_max_idx == expect_idx
}
