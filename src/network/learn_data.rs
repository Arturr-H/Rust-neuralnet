/// This struct is stored in each `Layer`, and is useful
/// for calculating partial derivatives in the backpropagation step 
pub struct LearnData {
    /// The Inputs which this layer recieved during forward prop
    /// 
    /// `len = size(L-1)`
    pub inputs: Vec<f64>,

    /// The "weighted neuron values" of the current layer. By that
    /// I mean the sum of each previous neuron's weights times their
    /// activation plus a bias stored here.
    /// 
    /// `len = size(L)`
    pub weighted_inputs: Vec<f64>,

    /// Same as the weighted inputs but after having the activation
    /// function applied.
    /// 
    /// `len = size(L-1)`
    pub activations: Vec<f64>,

    /// Stored partial derivatives for each neuron
    pub node_values: Vec<f64>
}

impl LearnData {
    pub fn new(current_layer_size: usize, prev_layer_size: usize) -> Self {
        Self {
            inputs: vec![0.0; prev_layer_size],
            weighted_inputs: vec![0.0; current_layer_size],
            activations: vec![0.0; current_layer_size],
            node_values: vec![0.0; current_layer_size]
        }
    }
}
