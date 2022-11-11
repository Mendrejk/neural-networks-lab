use crate::config::STANDARD_DISTRIBUTION;
use crate::{ActivationFunction, BATCH_SIZE};
use ndarray::{Array1, Array2, Axis};
use rand::distributions::Distribution;

pub struct BatchedNeuralLayer {
    pub weights: Array2<f64>,
    pub(crate) biases: Array1<f64>,
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

        let mut weights = Array2::zeros((neuron_count, input_size));
        for y in 0..neuron_count {
            for x in 0..input_size {
                weights[[y, x]] = normal_distribution.sample(&mut rng) as f64;
            }
        }

        let biases = Array1::from(
            (0..neuron_count)
                .map(|_| normal_distribution.sample(&mut rng) as f64)
                .collect::<Vec<f64>>(),
        );

        Self {
            weights,
            biases,
            activation_function,
            stimuli: None,
            errors: None,
            neuron_count,
        }
    }

    pub fn calculate(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let stimuli = &self.weights.dot(inputs) + &self.biases;
        stimuli.mapv(|stimulus| self.activation_function.calculate(stimulus))
    }

    pub fn calculate_batch(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut stimuli = Array2::zeros((self.neuron_count, BATCH_SIZE));
        for (batch_index, mut batch_stimuli_row) in stimuli.axis_iter_mut(Axis(1)).enumerate() {
            batch_stimuli_row.assign(
                &(&self.weights.dot(&inputs.index_axis(Axis(1), batch_index)) + &self.biases),
            );
        }

        let activations = stimuli.mapv(|stimulus| self.activation_function.calculate(stimulus));

        self.stimuli = Some(stimuli);
        activations
    }

    pub fn calculate_errors(
        &self,
        next_weights: &Array2<f64>,
        next_errors: &Array2<f64>,
    ) -> Array2<f64> {
        let mut next_res = Array2::zeros((self.neuron_count, BATCH_SIZE));
        for (batch_index, mut batch_next_res_row) in next_res.axis_iter_mut(Axis(1)).enumerate() {
            batch_next_res_row.assign(
                &next_weights
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
