use serde_derive::{Deserialize, Serialize};




/// What activation function to use
pub struct NetworkActivations {
    pub activation: ActivationType,
    pub output_activation: ActivationType
}
#[derive(Clone, Copy)]
pub struct Activation {
    pub function: fn(&Vec<f64>, usize) -> f64,
    pub derivative: fn(&Vec<f64>, usize) -> f64
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[repr(u8)]
pub enum ActivationType {
    Relu, LeakyRelu, Sigmoid, Softmax
}
impl ActivationType {
    pub fn into_activation(&self) -> Activation {
        match &self {
            Self::LeakyRelu => Activation::leaky_relu(),
            Self::Relu => Activation::relu(),
            Self::Sigmoid => Activation::sigmoid(),
            Self::Softmax => Activation::softmax(),
        }
    }
}

impl NetworkActivations {
    pub fn new(activation: ActivationType, output_activation: ActivationType) -> Self {
        Self { activation, output_activation }
    }
}
impl Activation {
    pub fn relu() -> Self {
        Self {
            function: relu,
            derivative: relu_derivative
        }
    }
    pub fn leaky_relu() -> Self {
        Self {
            function: leaky_relu,
            derivative: leaky_relu_derivative
        }
    }
    pub fn sigmoid() -> Self {
        Self {
            function: sigmoid,
            derivative: sigmoid_derivative
        }
    }
    pub fn softmax() -> Self {
        Self {
            function: softmax,
            derivative: softmax_derivative
        }
    }
}

pub fn relu(inputs: &Vec<f64>, index: usize) -> f64 {
    if inputs[index] > 0.0 { return inputs[index] }
    else { return 0.0 }
}
pub fn relu_derivative(inputs: &Vec<f64>, index: usize) -> f64 {
    (inputs[index] > 0.0) as u8 as f64
}

const LEAKY_RELU_NEGATIVE_SLOPE: f64 = 0.1;
pub fn leaky_relu(inputs: &Vec<f64>, index: usize) -> f64 {
    if inputs[index] > 0.0 { return inputs[index] }
    else { return LEAKY_RELU_NEGATIVE_SLOPE * inputs[index] }
}
pub fn leaky_relu_derivative(inputs: &Vec<f64>, index: usize) -> f64 {
    if inputs[index] > 0.0 { 1.0 } else { LEAKY_RELU_NEGATIVE_SLOPE }
}

pub fn sigmoid(inputs: &Vec<f64>, index: usize) -> f64 {
    1.0 / (1.0 + f64::exp(-inputs[index]))
}
pub fn sigmoid_derivative(inputs: &Vec<f64>, index: usize) -> f64 {
    let sig_x = sigmoid(inputs, index);
    sig_x * (1.0 - sig_x)
}

fn softmax(inputs: &Vec<f64>, index: usize) -> f64 {
    let mut exponent_sum = 0.0;
    for i in 0..inputs.len() {
        exponent_sum += inputs[i].exp();
    }

    let res = inputs[index].exp() / exponent_sum;

    return res;
}

fn softmax_derivative(inputs: &Vec<f64>, index: usize) -> f64 {
    let mut exponent_sum = 0.0;
    for i in 0..inputs.len() {
        exponent_sum += inputs[i].exp();
    }

    let ex = inputs[index].exp();
    return (ex * exponent_sum - ex * ex) / (exponent_sum * exponent_sum);
}

// for #[serde(skip)] macro
impl Default for Activation {
    fn default() -> Self {
        Self::leaky_relu()
    }
}
