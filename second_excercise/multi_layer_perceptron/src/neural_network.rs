use crate::config::{INPUT_SIZE, OUTPUT_SIZE};
use crate::learn_data::LearnData;
use crate::neural_layer::{ActivationFunction, NeuralLayer};
use crate::soft_max_layer::SoftMaxLayer;
use ndarray::Array1;

pub struct NeuralNetwork {
    neural_layers: Vec<NeuralLayer>,
    soft_max_layer: Option<SoftMaxLayer>,
}

impl NeuralNetwork {
    pub fn new(
        layer_sizes: Vec<usize>,
        activation_function: ActivationFunction,
        use_softmax: bool,
    ) -> Self {
        let mut neural_layers = vec![NeuralLayer::new(
            layer_sizes[0],
            INPUT_SIZE,
            activation_function,
        )];
        neural_layers.append(
            &mut (1..layer_sizes.len())
                .map(|i| NeuralLayer::new(layer_sizes[i], layer_sizes[i - 1], activation_function))
                .collect::<Vec<NeuralLayer>>(),
        );
        if !use_softmax {
            neural_layers.push(NeuralLayer::new(
                OUTPUT_SIZE,
                *layer_sizes.last().unwrap(),
                activation_function,
            ));
        }

        let soft_max_layer = match use_softmax {
            true => Some(SoftMaxLayer::new(OUTPUT_SIZE, *layer_sizes.last().unwrap())),
            false => None,
        };

        Self {
            neural_layers,
            soft_max_layer,
        }
    }

    pub fn calculate(&self, learn_data: &LearnData) -> bool {
        let mut result: Array1<f64> = learn_data.to_array();

        for layer in &self.neural_layers {
            result = layer.calculate(&result);
        }

        if let Some(soft_max_layer) = &self.soft_max_layer {
            result = soft_max_layer.calculate(&result);
        }

        println!("{:?}", result);

        let result_tuple =
            result
                .iter()
                .enumerate()
                .fold((0, result[0]), |(id_max, val_max), (id, val)| {
                    if &val_max > val {
                        (id_max, val_max)
                    } else {
                        (id, *val)
                    }
                });

        result_tuple.0 == learn_data.expected_class
    }
}
