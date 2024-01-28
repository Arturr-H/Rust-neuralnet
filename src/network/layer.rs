/* Imports */
use rand::thread_rng;
use rand::distributions::Distribution;
use rand_distr::Normal;
use serde_derive::{Deserialize, Serialize};
use crate::{activation::Activation, cost::Cost, learn_data::LearnData};

/// A neural network layer, storing weights, biases
/// and training data.
#[derive(Serialize, Deserialize)]
pub struct Layer {
    /// A two dimensional array containing each neurons
    /// weights. The dimensions are: `[[w; CURR_LAYER_SIZE]; PREV_LAYER_SIZE]`.
    /// 
    /// So the columns in that array depict each neuron in the previous layer, 
    /// and the rows are each weight which that neuron "owns". Therefore if
    /// you take weights[0..i][a] you'll get each weight which connects
    /// to the neuron with index a in the next layer.
    /// ? Todo: Make this one-dimensional for efficiency?
    pub weights: Vec<Vec<f64>>,
    pub cost_gradient_weights: Vec<Vec<f64>>,
    weight_velocities: Vec<Vec<f64>>,

    /// A one-dimensional array containing a bias
    /// for each neuron in the current layer
    /// ? Todo: Maybe look into having one bias for the entire layer?
    biases: Vec<f64>,
    cost_gradient_biases: Vec<f64>,
    bias_velocities: Vec<f64>,

    /// The activation function used for the current layer.
    #[serde(skip)] 
    activation: Activation,

    /// The amount of neurons in the current layer
    size: usize,
}

impl Layer {
    /// How do we want to initialize the network?
    /// 
    /// The weights need a good initialization method. If
    /// we provide a random value between 0 and 1 for every
    /// weight which is a bit too much, and if we're using
    /// the sigmoid activation funciton for our output layer,
    /// our gradients will vanish, becuase they are tending 
    /// towards 1. 
    /// 
    /// The same can be said if our weights are small, they
    /// will make the output layer produce a value close to 0.
    /// 
    /// I'll use the Kaiming He Initialization because I'm
    /// primarily using ReLU / leaky ReLU for activation fns
    pub fn new(prev_layer_size: usize, current_layer_size: usize, activation: Activation) -> Self {
        // ? TODO: Look into other bias init implementations
        let biases = vec![0.1; current_layer_size as usize];

        // If it's not the input layer
        let std = f64::sqrt(2.0 / prev_layer_size as f64);
        let normal_distribution = Normal::new(0.0, std).unwrap();
        let mut rng = thread_rng();

        let weights: Vec<Vec<f64>> = (0..prev_layer_size)
            .map(|_| {
                (0..current_layer_size)
                    .map(|_| normal_distribution.sample(&mut rng))
                    .collect()
            })
            .collect();

        return Self {
            biases,
            cost_gradient_biases: vec![0.0; current_layer_size],
            bias_velocities: vec![0.0; current_layer_size],

            weights,
            cost_gradient_weights: vec![vec![0.0; current_layer_size]; prev_layer_size],
            weight_velocities: vec![vec![0.0; current_layer_size]; prev_layer_size],

            activation,
            size: current_layer_size,
        }
    }

    /// A "single feed forward". Takes the input neurons and calculates
    /// the weighted inputs through the specified activation function.
    /// 
    /// This will not be executed on the input layer (unless explicitly
    /// done so obviously)
    /// 
    /// `save` param decides wether to store in learn data or not 
    /// * [Rewritten] ✅✅✅
    pub fn output(&self, learn_data: &mut LearnData, inputs: Vec<f64>, save: bool) -> Vec<f64> {
        if save { learn_data.inputs = inputs.clone(); }
        let mut output = self.biases.clone();

        // Each previous neuron iter
        for (index, weight_matrix) in self.weights.iter().enumerate() {
            // Each weight conencting to the next layer
            for (weight_index, weight) in weight_matrix.iter().enumerate() {
                output[weight_index] += inputs[index] * weight;
            }
        }

        // Store learn data
        for i in 0..self.size {
            let activated = (self.activation.function)(output[i]);
            if save {
                learn_data.weighted_inputs[i] = output[i];
                learn_data.activations[i] = activated;
            }

            // Apply activation function
            output[i] = activated;
        }

        output
    }

    /// Horrible name but it works.
    /// This method only mutates the learn_data.node_values, it
    /// doesn't return anything
    /// 
    /// * [Rewritten]
    pub (crate) fn calculate_node_values_output_layer(&self, learn_data: &mut LearnData, expected_outputs: Vec<f64>, cost: &Cost) -> () {
        let output_cost_deriv = (cost.derivative)(&learn_data.activations, &expected_outputs);
        for (index, node_value) in learn_data.node_values.iter_mut().enumerate() {
            let d_c_wrt_a = output_cost_deriv[index];
            let d_a_wrt_z = (self.activation.derivative)(learn_data.weighted_inputs[index]);
            *node_value = d_c_wrt_a * d_a_wrt_z;
        }
    }

    /// Once again horrible name. Does basically the same as 
    /// the other ugly-named method but other values for hidden layers.
    /// 
    /// `old_layer` and `old_node_values` are acutally the "next" layer but
    /// because we're backpropagating we name it old as we're reverse-itering
    /// 
    /// * [Rewritten] ✅✅✅
    pub (crate) fn calculate_node_values_hidden_layer(&self, learn_data: &mut LearnData, old_layer: &Layer, old_node_values: &Vec<f64>) -> () {
        for node_index in 0..self.size {
            let mut node_value = 0.0;

            for (weight_index, weight) in old_layer.weights[node_index].iter().enumerate() {
                node_value += weight * old_node_values[weight_index];
            }

            let a_wrt_z_deriv = node_value * (self.activation.derivative)(learn_data.weighted_inputs[node_index]);
            // TODO RESEARCH WHY NET PERFORMS BETTER WITHOUT a_wrt_z_deriv
            learn_data.node_values[node_index] = node_value;// * a_wrt_z_deriv;
        }
    }

    /// This will be called every iteration, and adds to the cost gradient matrices
    /// * [Rewritten] ✅✅✅
    pub fn update_gradients(&mut self, learn_data: &mut LearnData) -> () {
        // Iterates over previous neurons (w_mtx_idx = prev_neuron_idx)
        for (w_mtx_idx, w_mtx) in self.weights.iter().enumerate() {
            let input = learn_data.inputs[w_mtx_idx];

            // Iterates over each weight previous neuron "owns"
            // which connects to neuron in the current layer.
            for (w_index, w) in w_mtx.iter().enumerate() {
                let cost_wrt_weight_deriv = input * learn_data.node_values[w_index];
                self.cost_gradient_weights[w_mtx_idx][w_index] += cost_wrt_weight_deriv;
            }
        }

        for i in 0..self.size {
            self.cost_gradient_biases[i] += learn_data.node_values[i];
        }
    }

    /// Will also clear the gradients, applies the weight and
    /// bias gradients to the weights and biases respectivly
    /// 
    /// * [Rewritten]
    pub fn apply_gradients(&mut self, batch_size: usize, learn_rate: f64, momentum: f64, regularization: f64) -> () {
        let alpha = learn_rate / batch_size as f64;
        let weight_decay = 1.0 - regularization * alpha;

        for (w_mtx_idx, w_mtx) in self.weights.iter_mut().enumerate() {
            for (w_index, w) in w_mtx.iter_mut().enumerate() {
                let weight_gradient_curr = self.cost_gradient_weights[w_mtx_idx][w_index];

                // Get&set velocity
                let velocity = (self.weight_velocities[w_mtx_idx][w_index] * momentum) - (weight_gradient_curr * alpha);
                self.weight_velocities[w_mtx_idx][w_index] = velocity;

                // Apply & clear
                // println!("[{w_mtx_idx}] -= {}", weight_gradient_curr * alpha + velocity);
                // println!("L{} | {} - {}", self.index, format_f64(w), format_f64(&((*w * weight_decay + velocity) - *w)));
                *w = *w * weight_decay + velocity;
                self.cost_gradient_weights[w_mtx_idx][w_index] = 0.0;
            }
        }

        for i in 0..self.size {
            let bias_gradient_curr = self.cost_gradient_biases[i];

            let velocity = self.bias_velocities[i] * momentum - bias_gradient_curr * alpha;
            self.bias_velocities[i] = velocity;

            self.biases[i] += velocity;
            self.cost_gradient_biases[i] = 0.0;
        }
    }
}

/* Debug implementation */
impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "   ").unwrap();

        for (_, w_mtx) in self.weights.iter().enumerate() {
            write!(f, "[ ").unwrap();
            for (_, w) in w_mtx.iter().enumerate() {
                write!(f, "{} ", format!("{:.3}", w)).unwrap();
            }
            write!(f, "] ").unwrap();
        }

        write!(f, "( ").unwrap();
        for i in 0..self.size {
            write!(f, "{:.3} ", self.biases[i]).unwrap();
        }
        write!(f, ")").unwrap();
        
        std::fmt::Result::Ok(())
    }
}
