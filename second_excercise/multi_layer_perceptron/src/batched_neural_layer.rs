use crate::config::STANDARD_DISTRIBUTION;
use crate::{ActivationFunction, BATCH_SIZE};
use ndarray::{Array, Array1, Array2, Array3, Axis};
use rand::distributions::Distribution;
use std::ops::Index;

pub struct BatchedNeuralLayer {
    pub weights: Array3<f64>,
    biases: Array2<f64>,
    pub activation_function: ActivationFunction,
    pub stimuli: Option<Array2<f64>>,
    pub errors: Option<Array2<f64>>,
    neuron_count: usize,
}

impl BatchedNeuralLayer {
    pub fn new(
        neuron_count: usize,
        input_size: usize,
        activation_function: ActivationFunction,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let normal_distribution = rand_distr::Normal::new(0.0, STANDARD_DISTRIBUTION).unwrap();

        let mut weights = Array3::zeros((neuron_count, input_size, BATCH_SIZE));
        for y in 0..neuron_count {
            for x in 0..input_size {
                for z in 0..BATCH_SIZE {
                    weights[[y, x, z]] = normal_distribution.sample(&mut rng) as f64;
                }
            }
        }

        let mut biases = Array2::zeros((neuron_count, BATCH_SIZE));

        for y in 0..neuron_count {
            for x in 0..BATCH_SIZE {
                biases[[y, x]] = normal_distribution.sample(&mut rng) as f64;
            }
        }

        Self {
            weights,
            biases,
            activation_function,
            stimuli: None,
            errors: None,
            neuron_count,
        }
    }

    pub fn calculate(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut stimuli = Array2::zeros((self.neuron_count, BATCH_SIZE));
        for (batch_index, mut batch_stimuli_row) in stimuli.axis_iter_mut(Axis(1)).enumerate() {
            batch_stimuli_row.assign(
                &(&self
                    .weights
                    .index_axis(Axis(2), batch_index)
                    .dot(&inputs.index_axis(Axis(1), batch_index))
                    + &self.biases.index_axis(Axis(1), batch_index)),
            );
        }

        let activations = stimuli.mapv(|stimulus| self.activation_function.calculate(stimulus));

        self.stimuli = Some(stimuli);
        activations
    }

    pub fn calculate_errors(
        &self,
        next_weights: &Array3<f64>,
        next_errors: &Array2<f64>,
    ) -> Array2<f64> {
        let mut next_res = Array2::zeros((self.neuron_count, BATCH_SIZE));
        for (batch_index, mut batch_next_res_row) in next_res.axis_iter_mut(Axis(1)).enumerate() {
            batch_next_res_row.assign(
                &next_weights
                    .index_axis(Axis(2), batch_index)
                    .t()
                    .dot(&next_errors.index_axis(Axis(1), batch_index)),
            );
        }

        let activation_derivatives = self
            .stimuli
            .as_ref()
            .unwrap()
            .mapv(|x| self.activation_function.calculate_derivative(x));

        next_res * activation_derivatives
    }
}
