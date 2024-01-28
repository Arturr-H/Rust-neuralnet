
/// Rectified linear unit
#[allow(non_snake_case)]
pub fn ReLU(input: f64) -> f64 {
    if input > 0.0 { return input }
    else { return 0.0 }
}
#[allow(non_snake_case)]
pub fn ReLU_derivative(input: f64) -> f64 {
    (input > 0.0) as u8 as f64
}

/// Leaky rectified linear unit
#[allow(non_snake_case)]
pub fn leaky_ReLU(input: f64) -> f64 {
    if input > 0.0 { return input }
    else { return 0.01 * input }
}
#[allow(non_snake_case)]
pub fn leaky_ReLU_derivative(input: f64) -> f64 {
    if input > 0.0 { 1.0 } else { 0.01 }
}


// Sigmoid activation function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}
// Derivative of the sigmoid function
pub fn sigmoid_derivative(x: f64) -> f64 {
    let sig_x = sigmoid(x);
    sig_x * (1.0 - sig_x)
}

