/* Imports */
use serde_derive::{ Deserialize, Serialize };
use crate::{
    activation::NetworkActivations,
    cost::{Cost, CostType},
    handler::Datavis,
    layer::Layer,
    learn_data::LearnData,
    learn_rate::LearnRate,
    utils::{colorize, format_f64, stylize, Color, Style}
};

/// The network struct contains all layers, and
/// hyperparameters like learn_rate and momentum
#[derive(Deserialize, Serialize)]
pub struct Network {
    layers: Vec<Layer>,

    #[serde(skip)]
    layer_learn_data: Vec<LearnData>,
    cost: CostType,

    #[serde(skip)]
    learn_rate: LearnRate,
    batch_size: usize,
    momentum: f64,
    regularization: f64,

    #[serde(skip)]
    // Needs to be optional because fn(...) -> can't
    // implement the Default trait as far as I know.
    is_correct: Option<fn(&Vec<f64>, &Vec<f64>) -> bool>
}

impl Network {
    /// Documentation on how individual layers are initialized are
    /// located in `layer.rs`
    pub fn new<const N: usize>(
        layer_sizes: [usize; N],
        batch_size: usize,
        learn_rate: LearnRate,
        momentum: f64,
        regularization: f64,
        cost: CostType,
        activation: NetworkActivations,
        is_correct: fn(&Vec<f64>, &Vec<f64>) -> bool
    ) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        let mut layer_learn_data: Vec<LearnData> = Vec::new();

        for (layer_index, size) in layer_sizes.into_iter().enumerate() {
            // We won't treat the input layer as a `Layer`
            // because it's only input values.
            if layer_index == 0 { continue; }
            
            // Activation function
            let is_output_layer = layer_index == layer_sizes.len() - 1;
            let activation = if is_output_layer { activation.output_activation }
            else { activation.activation };

            layers.push(Layer::new(layer_sizes[layer_index - 1], size, activation));
            layer_learn_data.push(LearnData::new(size, layer_sizes[layer_index - 1]))
        }

        Self {
            layers, cost, layer_learn_data,
            learn_rate, batch_size, momentum,
            regularization, is_correct: Some(is_correct)
        }
    }

    /// Feed forward input neurons & save to learn data
    fn output_save(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut input = input.clone();

        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            input = layer.output(&mut self.layer_learn_data[layer_index], input, true);
        }

        input
    }

    /// Feed forward input neurons
    pub fn classify(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut input = input.clone();

        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            input = layer.output(&mut LearnData::new(0, 0), input, false);
        }

        input
    }

    /// Train the network on a dataset
    /// 
    /// WARNING please don't forget to set the input
    /// and output layer sizes according to your data
    pub fn train(&mut self, dataset: Vec<(Vec<f64>, Vec<f64>)>) -> () {
        let mut correct_p = vec![false; 100];
        let mut avg_cost = vec![1.0; 100];
        let mut iteration: usize = 0;
        let dataset_len = dataset.len();
        let batch_size = self.batch_size;

        'outer: loop {
            // Iterate over batches
            for inner in 0..batch_size {
                let (input, expected_output) = &dataset[iteration];

                let predicted_output = self.output_save(&input);
                let cost = (Cost::from_type(self.cost).function)(predicted_output.clone(), expected_output.clone());

                // Make learn rate a bit smaller for each iter
                self.learn_rate.set(iteration, dataset_len);
                self.update_gradients((&input, expected_output.clone()));

                iteration += 1;
                iteration_status_display(&self, iteration, self.learn_rate.get(), cost, &mut correct_p, &mut avg_cost, &predicted_output, &expected_output);

                if iteration > dataset_len - 1 {
                    break 'outer;
                }
            };

            // Apply gradients
            // println!("{}", colorize(Color::BrightGreen, "▶︎▶︎ Applying gradients ◀︎◀︎"));
            self.apply_gradients();
        }

        // Save network
        println!("{}", colorize(Color::Yellow, "▶︎▶︎ Saving network ◀︎◀︎"));
        self.save("./src/saves/network_v2");
    }

    pub fn update_gradients(&mut self, datapoint: (&Vec<f64>, Vec<f64>)) -> () {
        let output_layer_idx = self.layers.len() - 1;
        let output_layer = &mut self.layers[output_layer_idx];
        output_layer.calculate_node_values_output_layer(&mut self.layer_learn_data[output_layer_idx], datapoint.1, &&Cost::from_type(self.cost));
        output_layer.update_gradients(&mut self.layer_learn_data[output_layer_idx]);

        for layer_idx in (0..output_layer_idx).rev() {
            let prev_layer_index = layer_idx + 1;
            let prev_layer = &self.layers[prev_layer_index];

            // * TODO: Some workaround to skip clone...
            let old_node_values = &self.layer_learn_data[prev_layer_index].node_values.clone();

            self.layers[layer_idx].calculate_node_values_hidden_layer(&mut self.layer_learn_data[layer_idx], &prev_layer, old_node_values);
            self.layers[layer_idx].update_gradients(&mut self.layer_learn_data[layer_idx]);
        }
    }

    /// Also clears the gradients
    pub fn apply_gradients(&mut self) -> () {
        for layer in self.layers.iter_mut() {
            layer.apply_gradients(self.batch_size, self.learn_rate.get(), self.momentum, self.regularization);
        }
    }

    /// Write network to a save file for retrieval
    pub fn save(&self, path: &str) -> () {
        let serialized = match bincode::serialize(self) {
            Ok(e) => e,
            Err(e) => {
                println!("Could not serialize network {e}");
                return
            }
        };
        match std::fs::write(path, serialized) {
            Ok(_) => (),
            Err(e) => println!("Could not save network {e}")
        };
    }

    /// Get network from a save file
    pub fn retrieve_from_save(path: &str) -> Self {
        let bytes = match std::fs::read(path) {
            Ok(e) => e,
            Err(e) => panic!("Could not read network from file {e}")
        };

        match bincode::deserialize::<Self>(&bytes) {
            Ok(e) => e,
            Err(e) => panic!("Could not serialize network {e}")
        }
    }
}

/* Debug implementation */
impl std::fmt::Debug for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (layer_index, layer) in self.layers.iter().enumerate() {
            writeln!(f, "LAYER {} {:?} => {:?}", layer_index, layer, self.layer_learn_data[layer_index].activations).unwrap();
        }
        
        std::fmt::Result::Ok(())
    }
}

fn iteration_status_display(
    net: &Network,
    iteration: usize,
    learn_rate: f64,
    cost: f64,
    correct_p: &mut Vec<bool>,
    avg_cost: &mut Vec<f64>,
    predicted_output: &Vec<f64>,
    expected_output: &Vec<f64>
) -> () {
    avg_cost.rotate_left(1);
    avg_cost[99] = cost;

    let avg_cost_ = avg_cost.iter().sum::<f64>() / 100.0;
    let avg_cost_string = format!("{:.6}", avg_cost_);
    let avg_cost_color = if avg_cost_.abs() > 0.1 { Color::Red } else { Color::Blue };
    let is_correct = net.is_correct.unwrap()(predicted_output, expected_output);
    correct_p.rotate_left(1);
    correct_p[99] = is_correct;
    let avg_correct_ = correct_p.iter().map(|&e| e as u8).sum::<u8>() as f64 / 100.0;
    
    println!("{} {} {} {} {}",
        colorize(avg_cost_color, stylize(Style::Bold, stylize(Style::Reverse, " cost "))),
        format!("{}", colorize(avg_cost_color, stylize(Style::Bold, avg_cost_string))),
        colorize(avg_cost_color, stylize(Style::Reverse, format!("{:^5}", iteration))),

        format!("LR: {}", format_f64(&learn_rate)),
        format!("{:^5} {} [{}{}]",
            format!("{:.0}%", avg_correct_ * 100.0),
            if is_correct { "✅" } else { "🚫" },
            colorize(Color::Blue, "◼︎".repeat((50.0 * avg_correct_) as usize)),
            stylize(Style::Dim, colorize(Color::Black, "◼︎".repeat((50.0 * (1.0 - avg_correct_)) as usize)))
        )
    );
    // println!("[ {}] {} [ {}]      ô {} {}",
    //     colorize(Color::White, display_array(predicted_output)),
    //     colorize(Color::Blue, stylize(Style::Dim, "◼︎◼︎◼︎◼︎◼︎◼︎►")),
    //     colorize(Color::White, display_array(expected_output)),

    //     stylize(Style::Italic, stylize(Style::Bold, "mse")),
    //     predicted_output.iter().zip(expected_output).map(|(o, t)| {
    //         let diff = (o - t).powi(2);
    //         let col = if diff.abs() > ERR_DIFF_THRESH { Color::Red } else { Color::Blue };
    //         colorize(col, stylize(Style::Reverse, format!(" {:.3} ", diff)))
    //     }).collect::<String>()
    // );

    // println!("{:?}", net.layer_learn_data.iter().map(|e| &e.node_values).collect::<Vec<&Vec<f64>>>());
}
