


/// What activation function to use
pub struct NetworkActivations {
    pub activation: Activation,
    pub output_activation: Activation
}
#[derive(Clone, Copy)]
pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64
}

impl NetworkActivations {
    pub fn new(activation: Activation, output_activation: Activation) -> Self {
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
}

pub fn relu(input: f64) -> f64 {
    if input > 0.0 { return input }
    else { return 0.0 }
}
pub fn relu_derivative(input: f64) -> f64 {
    (input > 0.0) as u8 as f64
}

const LEAKY_RELU_NEGATIVE_SLOPE: f64 = 0.1;
pub fn leaky_relu(input: f64) -> f64 {
    if input > 0.0 { return input }
    else { return LEAKY_RELU_NEGATIVE_SLOPE * input }
}
pub fn leaky_relu_derivative(input: f64) -> f64 {
    if input > 0.0 { 1.0 } else { LEAKY_RELU_NEGATIVE_SLOPE }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}
pub fn sigmoid_derivative(x: f64) -> f64 {
    let sig_x = sigmoid(x);
    sig_x * (1.0 - sig_x)
}

// for #[serde(skip)] macro
impl Default for Activation {
    fn default() -> Self {
        Self::leaky_relu()
    }
}
