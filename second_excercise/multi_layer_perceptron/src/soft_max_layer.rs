use crate::config::STANDARD_DISTRIBUTION;
use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use std::f64::consts::E;

pub struct SoftMaxLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl SoftMaxLayer {
    pub fn new(neuron_count: usize, input_size: usize) -> Self {
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

        Self { weights, biases }
    }

    pub fn calculate(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let stimuli = &self.weights.dot(inputs) + &self.biases;

        let e_values = stimuli.mapv(|stimulus| E.powf(stimulus));
        let e_sum = e_values.sum();

        Array1::from(
            stimuli
                .iter()
                .enumerate()
                .map(|(index, _)| e_values[index] / e_sum)
                .collect::<Vec<f64>>(),
        )
    }
}
