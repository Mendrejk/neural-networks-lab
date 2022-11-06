use crate::learn_data::LearnData;
use crate::neural_layer::ActivationFunction;
use crate::neural_network::NeuralNetwork;

mod config;
mod learn_data;
mod neural_layer;
mod neural_network;
mod soft_max_layer;

fn main() {
    let (train_data, test_data) = LearnData::load_mnist();
    let network = NeuralNetwork::new(vec![5, 3, 2], ActivationFunction::HiperbolicTangent, false);
    println!("{}", network.calculate(train_data.first().unwrap()));
}
