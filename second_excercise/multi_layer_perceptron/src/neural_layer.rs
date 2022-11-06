use crate::config::STANDARD_DISTRIBUTION;
use std::f64::consts::E;

use ndarray::{Array1, Array2};
use rand::distributions::Distribution;

#[derive(Copy, Clone)]
pub enum ActivationFunction {
    Sigmoidal,
    HiperbolicTangent,
    RectifiedLinearUnit,
}

impl ActivationFunction {
    pub fn calculate(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoidal => Self::sigmoidal(x),
            Self::HiperbolicTangent => Self::hiperbolic_tangent(x),
            Self::RectifiedLinearUnit => Self::rectified_linear_unit(x),
        }
    }

    pub fn calculate_derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoidal => {
                let result = Self::sigmoidal(x);
                result * (1. - result)
            }
            ActivationFunction::HiperbolicTangent => {
                let result = Self::hiperbolic_tangent(x);
                1. - (result * result)
            }
            // Using softplus's derivative (so the Sigmoidal) as an approximation
            ActivationFunction::RectifiedLinearUnit => Self::sigmoidal(x),
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
    pub weights: Array2<f64>,
    biases: Array1<f64>,
    pub activation_function: ActivationFunction,
    pub stimuli: Option<Array1<f64>>,
    pub errors: Option<Array1<f64>>,
}

impl NeuralLayer {
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
        }
    }

    pub fn calculate(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let b = self.weights.as_slice();
        let stimuli = &self.weights.dot(inputs) + &self.biases;

        let activations = stimuli.mapv(|stimulus| self.activation_function.calculate(stimulus));

        self.stimuli = Some(stimuli);
        activations
    }

    pub fn calculate_errors(
        &self,
        next_weights: &Array2<f64>,
        next_errors: &Array1<f64>,
    ) -> Array1<f64> {
        let next_res = next_weights.t().dot(next_errors);
        let activation_derivatives = self
            .stimuli
            .as_ref()
            .unwrap()
            .mapv(|x| self.activation_function.calculate_derivative(x));

        next_res * activation_derivatives
    }
}
