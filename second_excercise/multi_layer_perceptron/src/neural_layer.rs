use crate::config::STANDARD_DISTRIBUTION;
use crate::neuron::Neuron;

use ndarray::{arr2, array, Array, Array1, Array2, Ix1, Ix2, ShapeBuilder};
use rand::distributions::Distribution;

pub struct NeuralLayer<'a> {
    neurons: Array1<Neuron<'a>>,
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl<'a> NeuralLayer<'a> {
    pub fn new(neuron_count: usize, input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal_distribution = rand_distr::Normal::new(0.0, STANDARD_DISTRIBUTION).unwrap();

        let neurons: Array1<Neuron> = Array1::from(
            (0..neuron_count)
                .map(|_| Neuron::new())
                .collect::<Vec<Neuron>>(),
        );
        let mut weights = Array2::zeros((neuron_count, input_size));
        for y in 0..neuron_count {
            for x in 0..input_size {
                weights[[y, x]] = normal_distribution.sample(&mut rng) as f64;
            }
        }
        let biases = Array1::from(
            (0..input_size)
                .map(|_| normal_distribution.sample(&mut rng) as f64)
                .collect::<Vec<f64>>(),
        );

        Self {
            neurons,
            weights,
            biases,
        }
    }
}
