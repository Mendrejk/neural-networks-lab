use crate::config::STANDARD_DISTRIBUTION;
use crate::BATCH_SIZE;
use ndarray::{Array1, Array2, Array3, Axis};
use rand::distributions::Distribution;
use std::f64::consts::E;

pub struct BatchedSoftMaxLayer {
    pub weights: Array3<f64>,
    biases: Array2<f64>,
    pub stimuli: Option<Array2<f64>>,
    neuron_count: usize,
}

impl BatchedSoftMaxLayer {
    pub fn new(neuron_count: usize, input_size: usize) -> Self {
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
            stimuli: None,
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

        let e_values = stimuli.mapv(|stimulus| E.powf(stimulus));
        self.stimuli = Some(stimuli);

        let e_sums = Array1::from(
            e_values
                .axis_iter(Axis(1))
                .map(|e_values_col| e_values_col.sum())
                .collect::<Vec<f64>>(),
        );

        let mut activations = Array2::zeros((self.neuron_count, BATCH_SIZE));
        for y in 0..self.neuron_count {
            for x in 0..BATCH_SIZE {
                activations[[y, x]] = e_values[[y, x]] / e_sums[y];
            }
        }

        activations
    }

    pub fn calculate_activation_derivative(&self, x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x)) // TODO - sigmoidal derivative is used here
    }
}
