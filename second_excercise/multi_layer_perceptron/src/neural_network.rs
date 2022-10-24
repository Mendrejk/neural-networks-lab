use crate::config::{INPUT_SIZE, OUTPUT_SIZE};
use crate::neural_layer::{ActivationFunction, NeuralLayer};
use crate::soft_max_layer::SoftMaxLayer;

pub struct NeuralNetwork {
    neural_layers: Vec<NeuralLayer>,
    soft_max_layer: Option<SoftMaxLayer>,
}

impl NeuralNetwork {
    pub fn new(
        inner_layer_count: usize,
        layer_size: usize,
        activation_function: ActivationFunction,
        use_softmax: bool,
    ) -> Self {
        let mut neural_layers = vec![NeuralLayer::new(
            layer_size,
            INPUT_SIZE,
            activation_function,
        )];
        neural_layers.append(
            &mut (0..inner_layer_count)
                .map(|_| NeuralLayer::new(layer_size, layer_size, activation_function))
                .collect::<Vec<NeuralLayer>>(),
        );

        let soft_max_layer = match use_softmax {
            true => Some(SoftMaxLayer::new(OUTPUT_SIZE, layer_size)),
            false => None,
        };

        Self {
            neural_layers,
            soft_max_layer,
        }
    }
}
