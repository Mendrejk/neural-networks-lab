use crate::config::BATCH_SIZE;
use crate::learn_data::LearnData;
use crate::neural_layer::ActivationFunction;
use crate::neural_network::NeuralNetwork;

mod batched_neural_layer;
mod batched_soft_max_layer;
mod config;
mod learn_data;
mod neural_layer;
mod neural_network;
mod soft_max_layer;

fn main() {
    let (train_data, test_data) = LearnData::load_mnist();
    let mut network =
        NeuralNetwork::new(vec![5, 3, 2], ActivationFunction::HiperbolicTangent, false);
    // println!("{}", network.calculate(train_data.first().unwrap()));

    if let Some(batch) = train_data.chunks(BATCH_SIZE).next() {
        network.learn_batch(batch);
    }
}
