use crate::config::STANDARD_DISTRIBUTION;
use std::f64::consts::E;

use ndarray::{arr2, array, Array, Array1, Array2, Ix1, Ix2, ShapeBuilder};
use rand::distributions::Distribution;

enum ActivationFunction {
    Sigmoidal,
    HiperbolicTangent,
    RectifiedLinearUnit,
}

impl ActivationFunction {
    pub fn calculate(self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoidal => ActivationFunction::sigmoidal(x),
            ActivationFunction::HiperbolicTangent => ActivationFunction::hiperbolic_tangent(x),
            ActivationFunction::RectifiedLinearUnit => ActivationFunction::rectified_linear_unit(x),
        }
    }

    fn sigmoidal(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn hiperbolic_tangent(x: f64) -> f64 {
        2.0 / (1.0 + E.powf(-2.0 * x)) - 1.0
    }

    fn rectified_linear_unit(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }
}

pub struct NeuralLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl NeuralLayer {
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
            (0..input_size)
                .map(|_| normal_distribution.sample(&mut rng) as f64)
                .collect::<Vec<f64>>(),
        );

        Self { weights, biases }
    }
}
