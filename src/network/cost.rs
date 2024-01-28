use serde_derive::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Copy)]
pub enum CostType {
    MSE,
    CrossEntropy
}

/// What cost function to use.
pub(crate) struct Cost {
    pub function: fn(Vec<f64>, Vec<f64>) -> f64,
    pub derivative: fn(&Vec<f64>, &Vec<f64>) -> Vec<f64>
}

impl Cost {
    pub fn from_type(type_: CostType) -> Self {
        match type_ {
            CostType::CrossEntropy => Cost::cross_entropy(),
            CostType::MSE => Cost::mse()
        }
    }
    pub fn mse() -> Self {
        Self {
            function: mse,
            derivative: mse_derivative
        }
    }
    pub fn cross_entropy() -> Self {
        Self {
            function: cross_entropy,
            derivative: cross_entropy_derivative
        }
    }
}

fn mse(predicted_outputs: Vec<f64>, expected_outputs: Vec<f64>) -> f64 {
    let mut cost = 0.0;
    for i in 0..predicted_outputs.len() {
        let error = predicted_outputs[i] - expected_outputs[i];
        cost += error * error;
    }
    0.5 * cost
}

fn mse_derivative(predicted_output: &Vec<f64>, expected_output: &Vec<f64>) -> Vec<f64> {
    predicted_output.into_iter().zip(expected_output).map(|(p, e)| p - e).collect()
}

/// SUPER WARRNiNG: THIS FUNCTION REQUIRES ALL EXPECTED OUTPUTS
/// TO BE EITHER 0 OR 1 SO NO OUTPUT RANDOM NOISE PLEASE
fn cross_entropy(predicted_outputs: Vec<f64>, expected_outputs: Vec<f64>) -> f64 {
    let mut cost = 0.0;
    for i in 0..predicted_outputs.len() {
        let x = predicted_outputs[i];
        let y = expected_outputs[i];
        let v = if y == 1.0 { - x.ln() } else { (1.0 - x).ln() };
        cost += if v.is_nan() { 0.0 } else { v };
    }

    return cost;
}

fn cross_entropy_derivative(predicted_outputs: &Vec<f64>, expected_outputs: &Vec<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(expected_outputs.len());
    for i in 0..expected_outputs.len() {
        let x = predicted_outputs[i];
        let y = expected_outputs[i];
        if x == 0.0 || x == 1.0 {
            out.push(0.0);
        };
        out.push((-x + y) / (x * (x - 1.0)));
    }

    out
}

// For the #[serde(skip)] macro
impl Default for Cost {
    fn default() -> Self {
        Self::mse()
    }
}
