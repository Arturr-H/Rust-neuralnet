/* Imports */
use rand::{ self, Rng };
use crate::{ activation::{ leaky_ReLU, leaky_ReLU_derivative, sigmoid, sigmoid_derivative }, utils::log };

const GRADIENT_CLIP_ROOF: f64 = 0.001f64;
const GRADIENT_CLIP_FLOOR: f64 = 10f64;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Network {
    /// The layer sizes, first item is input layer
    /// size, last one is output layer size.
    layer_sizes: Vec<usize>,

    /// Biases for individual neurons
    biases: Vec<Vec<f64>>,

    /// ## The weights matrix - 3d
    /// - The first `Vec` indicates all layers
    /// - The second vector indicates all neurons in
    /// a layer 
    /// - The third vector indicates each neurons weights.
    /// Therefore indexing the third vector will yield a
    /// single weight.
    weights: Vec<Vec<Vec<f64>>>,

    /// The learning rate used to determening slope steps
    /// in gradient descent.
    learning_rate: f64,

    /// A hyperparameter like the learning_rate, it's a value
    /// around 0 - 1, commonly closer to 1. It's multiplied
    /// by the previous gradient of a weight and added together
    /// with the current gradient of a weight to keep some momentum
    /// 
    /// Keep in mind that it can be applied likewise for biases
    momentum: f64,

    /// A hyperparameter handling penalty for overfitting
    regularization: f64,

    /// From doing feed-forward we often want to save
    /// the neurons weighted inputs for calculating some 
    /// partial derivatives for the cost function.
    /// (***pre*** activation function)
    weighted_inputs: Vec<Vec<f64>>,

    /// This 2d-matrix is used in backpropagation
    /// and stores partial derivatives, used later
    /// for applying to gradients
    node_values: Vec<Vec<f64>>,

    /// The networks activation function
    activation_func: fn(f64) -> f64,
    activation_func_derivative: fn(f64) -> f64,

    cost_gradient_weights: Vec<Vec<Vec<f64>>>,
    cost_gradient_biases: Vec<Vec<f64>>,

    weight_velocities: Vec<Vec<Vec<f64>>>,
    bias_velocities: Vec<Vec<f64>>,
}

impl Network {
    /// I'll be using Kaiming initialization for
    /// intiailizing the weights.
    /// 
    /// Kaiming initialization is uses a normal distribution
    /// with mean 0 and variance 2/n (spread).
    pub fn new_kaiming_init(layer_sizes: Vec<usize>, learning_rate: f64, momentum: f64, regularization: f64) -> Self {
        let mut rng = rand::thread_rng();

        // TODO: Check for better methods than just initializing
        // TODO: all biases to 0.1 - might work though.
        let mut biases: Vec<Vec<f64>> = Vec::with_capacity(layer_sizes.len());
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(layer_sizes.len());

        // Each count of neurons of each layer.
        for layer_index in 0..(layer_sizes.len() - 1) {
            let layer_size = layer_sizes[layer_index];
            let next_layer_size = layer_sizes[layer_index + 1];

            // TODO: This might have to be prev_layer_size but I'm not fully sure
            // TODO: but it's probably not as important because this is the init function
            let gain = (2.0f64 / layer_size as f64).sqrt();

            // Each neuron has their own weights vector.
            let mut neurons: Vec<Vec<f64>> = Vec::with_capacity(layer_size);

            // Each neuron of each layer
            for _ in 0..layer_size {
                let mut w = vec![0.0f64; next_layer_size];
                w.fill_with(|| rng.gen::<f64>() * gain);
                neurons.push(w);
            }

            weights.push(neurons);
        }

        // INitialize biases
        for layer_size in &layer_sizes {
            biases.push(vec![0.1; *layer_size]);
        }

        Self {
            // I hate this so much but it's only for initialization so i don't really care 8)
            cost_gradient_weights: weights.clone().iter().map(|e| e.iter().map(|a| a.iter().map(|_| 0.0f64).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>()).collect::<Vec<Vec<Vec<f64>>>>(),
            cost_gradient_biases: biases.clone().iter().map(|e| e.iter().map(|_| 0.0f64).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>(),
            weight_velocities: weights.clone().iter().map(|e| e.iter().map(|a| a.iter().map(|_| 0.0f64).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>()).collect::<Vec<Vec<Vec<f64>>>>(),
            bias_velocities: biases.clone().iter().map(|e| e.iter().map(|_| 0.0f64).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>(),

            layer_sizes: layer_sizes.clone(),
            biases,
            weights,
            learning_rate,
            momentum,
            regularization,
            weighted_inputs: layer_sizes.iter().map(|e| vec![0.0; *e]).collect(),
            node_values: layer_sizes.iter().map(|e| vec![0.0; *e]).collect(),
            activation_func: leaky_ReLU,
            activation_func_derivative: leaky_ReLU_derivative,
        }
    }

    /// Private method, does the same as `feed_forward` would do,
    /// but saves the neuron activations to `weighted_inputs`
    /// field and requires a mutable reference to self.
    pub fn feed_forward_save(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut inputs = inputs.clone();

        for layer_index in 1..self.layer_sizes.len() {
            let nodes_in_curr_layer = self.layer_sizes[layer_index];
            let nodes_in_prev_layer = self.layer_sizes[layer_index - 1];

            /* Feed the outputs of the previous layer
                into the next layer as inputs */
            inputs = {
                let mut output = vec![0f64; nodes_in_curr_layer];

                for node_out in 0..nodes_in_curr_layer {
                    let mut base = self.biases[layer_index][node_out];
        
                    for node_in in 0..nodes_in_prev_layer {
                        // println!("node_in {node_in}");
                        let prev_activation = inputs[node_in];
                        let weight = self.weights[layer_index - 1][node_in][node_out];
                        base += prev_activation * weight;
                    };

                    /* Activation function */
                    output[node_out] = (self.activation_func)(base);
                };
        
                output
            };

            self.weighted_inputs[layer_index] = inputs.clone();
        };

        inputs
    }

    pub fn train(&mut self, data: Vec<(Vec<f64>, Vec<f64>)>) -> () {
        let mut iteration = 1;
        let mut best_avg_batch_err = f64::NAN;
        let mut current_batch_err = 0.0;
        let last_layer_size = self.layer_sizes.last().unwrap().clone();

        log("backprop", "\n\nStarting training\n\n");
        for (input, target) in data.iter() {

            // We need to use the `feed_forward_save` method
            // here instead of the `feed_forward` because
            // the activations need to be saved for backprop
            let output = self.feed_forward_save(input);

            /* Logging */
            let err = Self::calculate_error(&output, target);
            let err_visual = err.iter().sum::<f64>() / last_layer_size as f64;
            current_batch_err += err_visual;


            //println!("{} \n {target:?} \n\n\n", output.iter().map(|e| format!("{:.2}", e)).collect::<Vec<String>>().join(", "));

            /* Backprop */
            self.backward_pass(&output, target);
            
            let batch_size = 4;
            if iteration % batch_size == 0 {
                // Error seems to go down quicker if we have a bigger batch size?
                self.apply_gradients(batch_size * 4);

                if current_batch_err < best_avg_batch_err || best_avg_batch_err.is_nan() {
                    best_avg_batch_err = current_batch_err;
                }

                /* Logging */
                log(
                    &format!("backprop i ={: ^6}", iteration.to_string()),
                    format!(
                        // "{:?}", self.weights[2][1]
                        "E ~ bs={batch_size}: {:<14} learn: {:.10}",
                        format!("{:.5}", current_batch_err),
                        self.learning_rate
                    )
                );

                if current_batch_err < 0.6 {
                    self.learning_rate = 0.00000002;
                }

                current_batch_err = 0.0;
            }

            iteration += 1;
        }
    }

    /// Get the activation of any layer (***PRE*** activation function) which
    /// is saved by the `feed_forward_save` method.
    fn layer_output(&self, index: usize) -> &Vec<f64> {
        &self.weighted_inputs[index] // TODO: CHECK!
    }
    fn calculate_error(output: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
        output.iter().zip(target).map(|(o, t)| (*o - *t).powi(2)).collect()
    }

    /// Will modify the `node_values` field, and update
    /// the `cost_gradient_weights` and `cost_gradient_biases`
    /// via the `update_gradients` method.
    fn backward_pass(&mut self, output: &Vec<f64>, target: &Vec<f64>) {
        // TODO don't clone
        let mut self_static = self.clone();

        // Iterate over all layers
        // TODO: Check we might need to start at 1, but maybe 0
        for layer_index in (1..self.layer_sizes.len()).rev() {
            let is_output_layer = layer_index == self.layer_sizes.len() - 1;

            // Iterate over each layers neurons
            for neuron_index in 0..self.layer_sizes[layer_index] {

                // NODE VALUES OUTPUT
                if is_output_layer {
                    let error_derivative: Vec<f64> = output.iter().zip(target).map(|(o, t)| 2.0 * (o - t)).collect();
                    let node_cost_deriv = 2.0 * error_derivative[neuron_index];
                    let activation_deriv = (self.activation_func_derivative)(self_static.layer_output(layer_index)[neuron_index]);

                    // We don't need to clear the `self.node_values` because we override here
                    self.node_values[layer_index][neuron_index] = node_cost_deriv * activation_deriv;
                }

                // NODE VALUES HIDDEN
                else {
                    let mut node_value = 0.0;

                    // The node value for this neuron is the sum of all
                    // the weights which this neurons has outgoing
                    // multiplied by the next node values (of the next layer)
                    //println!("ITERATING weig_l(): {}", self.weights[layer_index][neuron_index].len());
                    for (weight_index, weight) in self.weights[layer_index][neuron_index].iter().enumerate() {
                        node_value += self.node_values[layer_index + 1][weight_index] * weight;
                    }

                    // Then we need to multiply the node value by
                    // the derivative of the activation function
                    // with respect to the weighted input (z) 
                    node_value *= (self.activation_func_derivative)(self.weighted_inputs[layer_index][neuron_index]);
                
                    // We don't need to clear the `self.node_values` because we override here
                    self.node_values[layer_index][neuron_index] = node_value;
                }
            }
        }

        // After iterating over each layer and recieving new node_value:s,
        // we'll apply that in the gradient descent step here
        self.update_gradients();
    }

    /// Will be called after finishing a backward pass
    fn update_gradients(&mut self) -> () {
        // Iterate over all layers except input
        for layer_index in 1..self.layer_sizes.len() {
            let layer_size = self.layer_sizes[layer_index];

            // The `weight_matrix_index` is the same as
            // the previous neuron index, and `weight_index`
            // should be the same as the curr layer neuron index
            for (weight_matrix_index, weight_matrix) in self.weights[layer_index - 1].iter().enumerate() {
                for (weight_index, _) in weight_matrix.iter().enumerate() {
                    // The weight index here is connected
                    // to the neuron in the current layer
                    let node_values = self.node_values[layer_index][weight_index];
                    let prev_activations = self.weighted_inputs[layer_index - 1][weight_matrix_index];

                    self.cost_gradient_weights[layer_index - 1][weight_matrix_index][weight_index] += prev_activations * node_values;
                }
            }

            for i in 0..layer_size {
                // apply biases gradients, will be averaged later
                self.cost_gradient_biases[layer_index][i] += self.node_values[layer_index][i];
            }
        }
    }

    /// Will be called every `i % batch_size == 0` iterations.
    fn apply_gradients(&mut self, batch_size: usize) -> () {
        let learning_rate = self.learning_rate / batch_size as f64;
        let weight_decay = /*1.0 - self.regularization * */learning_rate;

        for (layer_index, layer) in self.weights.iter_mut().enumerate() {
            for (node_index, node) in layer.iter_mut().enumerate() {

                // apply weights
                for (weight_index, weight) in node.iter_mut().enumerate() {
                    let weight_gradient_curr = self.cost_gradient_weights[layer_index][node_index][weight_index];
                    
                    // Get&set velocity                    
                    let velocity = (self.weight_velocities[layer_index][node_index][weight_index] * self.momentum) - (weight_gradient_curr * learning_rate);
                    self.weight_velocities[layer_index][node_index][weight_index] = velocity;

                    *weight -= weight_gradient_curr * learning_rate;

                    // Reset cost gradient
                    self.cost_gradient_weights[layer_index][node_index][weight_index] = 0.0;
                }

                // Get&set velocity
                let bias_gradient_curr = self.cost_gradient_biases[layer_index][node_index];
                let velocity = self.bias_velocities[layer_index][node_index] * self.momentum - bias_gradient_curr * learning_rate;
                self.bias_velocities[layer_index][node_index] = velocity;

                // apply biases
                self.biases[layer_index][node_index] -= bias_gradient_curr * learning_rate;

                // Reset cost gradient
                self.cost_gradient_biases[layer_index][node_index] = 0.0;
            }
        }
    }
}
