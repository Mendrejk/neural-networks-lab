use crate::batched_neural_layer::BatchedNeuralLayer;
use crate::batched_soft_max_layer::BatchedSoftMaxLayer;
use crate::config::{BATCH_SIZE, IMAGE_SIZE, OUTPUT_SIZE, UPDATE_FACTOR};
use crate::learn_data::LearnData;
use crate::neural_layer::ActivationFunction;

use ndarray::{Array2, Axis};
use rand::seq::SliceRandom;

pub struct NeuralNetwork {
    neural_layers: Vec<BatchedNeuralLayer>,
    soft_max_layer: Option<BatchedSoftMaxLayer>,
}

impl NeuralNetwork {
    pub fn new(
        layer_sizes: Vec<usize>,
        activation_function: ActivationFunction,
        use_softmax: bool,
    ) -> Self {
        let mut neural_layers = vec![BatchedNeuralLayer::new(
            layer_sizes[0],
            IMAGE_SIZE,
            activation_function,
        )];
        neural_layers.append(
            &mut (1..layer_sizes.len())
                .map(|i| {
                    BatchedNeuralLayer::new(layer_sizes[i], layer_sizes[i - 1], activation_function)
                })
                .collect::<Vec<BatchedNeuralLayer>>(),
        );
        if !use_softmax {
            neural_layers.push(BatchedNeuralLayer::new(
                OUTPUT_SIZE,
                *layer_sizes.last().unwrap(),
                activation_function,
            ));
        }

        let soft_max_layer = match use_softmax {
            true => Some(BatchedSoftMaxLayer::new(
                OUTPUT_SIZE,
                *layer_sizes.last().unwrap(),
            )),
            false => None,
        };

        Self {
            neural_layers,
            soft_max_layer,
        }
    }

    pub fn calculate(&mut self, learn_data: &LearnData) -> (bool, f64) {
        let mut result = learn_data.to_neural_input();

        for layer in &mut self.neural_layers {
            result = layer.calculate(&result);
        }

        if let Some(soft_max_layer) = &mut self.soft_max_layer {
            result = soft_max_layer.calculate(&result);
        }

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

        let is_correct = result_tuple.0
            == learn_data
                .expected_class
                .iter()
                .position(|&elem| elem == 1)
                .unwrap();
        let likelihood =
            (-1.0 * result.mapv(|x| x.ln()) * &learn_data.expected_class.mapv(|x| x as f64)).sum();

        (is_correct, likelihood)
    }

    pub fn learn_batch(&mut self, learn_batch: &[LearnData]) {
        let mut results = Array2::zeros((IMAGE_SIZE, BATCH_SIZE));
        for (batch_index, mut batch_result_row) in results.axis_iter_mut(Axis(1)).enumerate() {
            batch_result_row.assign(&learn_batch[batch_index].to_neural_input());
        }

        let input_stimuli = results.clone();

        for layer in &mut self.neural_layers {
            results = layer.calculate_batch(&results);
        }

        if let Some(soft_max_layer) = &mut self.soft_max_layer {
            results = soft_max_layer.calculate_batch(&results);
        }

        let mut out_deltas = Array2::zeros((OUTPUT_SIZE, BATCH_SIZE));
        for y in 0..OUTPUT_SIZE {
            for x in 0..BATCH_SIZE {
                out_deltas[[y, x]] = results[[y, x]] - learn_batch[x].expected_class[y] as f64;
            }
        }

        let out_derivatives = if let Some(soft_max_layer) = &self.soft_max_layer {
            soft_max_layer
                .stimuli
                .as_ref()
                .unwrap()
                .mapv(|x| soft_max_layer.calculate_activation_derivative(x))
        } else {
            let last_layer = self.neural_layers.last().unwrap();
            last_layer
                .stimuli
                .as_ref()
                .unwrap()
                .mapv(|x| last_layer.activation_function.calculate_derivative(x))
        };
        let out_errors = out_deltas * out_derivatives;

        if self.soft_max_layer.is_some() {
            self.soft_max_layer.as_mut().unwrap().errors = Some(out_errors.clone());
        } else {
            let last = self.neural_layers.len() - 1;
            self.neural_layers[last].errors = Some(out_errors.clone());
        }

        let last_index = if self.soft_max_layer.is_some() {
            self.neural_layers.len() - 1
        } else {
            self.neural_layers.len() - 2
        };

        let mut next_errors = &out_errors;
        let mut next_weights = if self.soft_max_layer.is_some() {
            self.soft_max_layer.as_ref().unwrap().weights.clone()
        } else {
            self.neural_layers.last().unwrap().weights.clone()
        };

        for i in (0..=last_index).rev() {
            let errors;
            {
                errors = Some(self.neural_layers[i].calculate_errors(&next_weights, next_errors));
            }
            self.neural_layers[i].errors = errors;
            next_errors = self.neural_layers[i].errors.as_ref().unwrap();
            next_weights = self.neural_layers[i].weights.clone();
        }

        let mut previous_stimuli = &input_stimuli;
        for layer in &mut self.neural_layers {
            layer.weights = &layer.weights
                - UPDATE_FACTOR
                    * layer
                        .errors
                        .as_ref()
                        .unwrap()
                        .dot(&previous_stimuli.clone().reversed_axes());
            previous_stimuli = layer.stimuli.as_ref().unwrap();

            layer.biases =
                &layer.biases - UPDATE_FACTOR * layer.errors.as_ref().unwrap().sum_axis(Axis(1));
        }
    }

    pub fn learn(&mut self, mut train_data: Vec<LearnData>, test_data: Vec<LearnData>) {
        let mut rng = rand::thread_rng();
        loop {
            train_data.shuffle(&mut rng);

            train_data
                .chunks(BATCH_SIZE)
                .for_each(|batch| self.learn_batch(batch));

            let (successes, likelihood) = test_data.iter().map(|data| self.calculate(data)).fold(
                (0, 0.0),
                |(correct, sum_likelihood), (current, likelihood)| {
                    let current_int = if current { 1 } else { 0 };
                    return (correct + current_int, sum_likelihood + likelihood);
                },
            );
            println!(
                "{}, {}",
                successes as f64 / test_data.len() as f64,
                likelihood
            );
        }
    }
}
