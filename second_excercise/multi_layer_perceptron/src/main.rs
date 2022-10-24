use crate::learn_data::LearnData;
use crate::neural_layer::ActivationFunction;
use crate::neural_network::NeuralNetwork;

mod config;
mod learn_data;
mod neural_layer;
mod neural_network;
mod soft_max_layer;

fn main() {
    let network = NeuralNetwork::new(vec![5, 3, 2], ActivationFunction::HiperbolicTangent, false);
    let data = LearnData {
        image_parts: vec![0.5],
        expected_class: 3,
    };
    println!("{}", network.calculate(&data));
}
