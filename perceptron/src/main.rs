use crate::data_entry::DataEntry;
use perceptron::Perceptron;

use std::fs;
use rand::Rng;
use crate::biased_perceptron::BiasedPerceptron;

mod biased_perceptron;
mod data_entry;
mod perceptron;

fn main() {
    let data_entries = read_data();

    // first_task(&data_entries);
    second_task(&data_entries);
}

fn first_task(data_entries: &Vec<DataEntry>) {
    let learn_factor = 0.01;

    for theta in [0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2] {
        let mut x_weights: Vec<f32> = vec![];
        let mut y_weights: Vec<f32> = vec![];
        let mut epochs: Vec<u32> = vec![];

        for _experiment_iter in 0..10 {
            let mut got_wrong_answer = true;
            let mut current_epochs = 0;

            let mut perceptron = Perceptron {
                x_weight: 0.01,
                y_weight: 0.01,
            };

            while got_wrong_answer {
                got_wrong_answer = false;

                for entry in data_entries {
                    if !perceptron.learn(entry, learn_factor, theta) {
                        got_wrong_answer = true;
                    }
                }

                current_epochs += 1;
            }

            epochs.push(current_epochs);
            x_weights.push(perceptron.x_weight);
            y_weights.push(perceptron.y_weight);
        }

        let avg_epochs = epochs.iter().sum::<u32>() as f32 / epochs.len() as f32;
        let avg_x_weight = x_weights.iter().sum::<f32>() / x_weights.len() as f32;
        let avg_y_weight = y_weights.iter().sum::<f32>() / y_weights.len() as f32;
        println!(
            "theta: {}, avg epochs: {}, x weight: {}, y weight: {}",
            theta, avg_epochs, avg_x_weight, avg_y_weight
        );
    }
}

fn second_task(data_entries: &Vec<DataEntry>) {
    let learn_factor = 0.01;
    let mut rng = rand::thread_rng();

    for weight_range in [
        -1.0..1.0,
        -0.8..0.8,
        -0.5..0.5,
        -0.2..0.2,
        -0.1..0.1,
        -0.05..0.05,
        -0.01..0.01,
    ] {
        let mut x_weights: Vec<f32> = vec![];
        let mut y_weights: Vec<f32> = vec![];
        let mut bias_weights: Vec<f32> = vec![];
        let mut epochs: Vec<u32> = vec![];

        for _experiment_iter in 0..10 {
            let mut got_wrong_answer = true;
            let mut current_epochs = 0;

            let mut perceptron = BiasedPerceptron {
                x_weight: rng.gen_range(weight_range.clone()),
                y_weight: rng.gen_range(weight_range.clone()),
                bias_weight: rng.gen_range(weight_range.clone())
            };

            while got_wrong_answer {
                got_wrong_answer = false;

                for entry in data_entries {
                    if !perceptron.learn(entry, learn_factor) {
                        got_wrong_answer = true;
                    }
                }

                current_epochs += 1;
            }

            epochs.push(current_epochs);
            x_weights.push(perceptron.x_weight);
            y_weights.push(perceptron.y_weight);
            bias_weights.push(perceptron.bias_weight);
        }

        let avg_epochs = epochs.iter().sum::<u32>() as f32 / epochs.len() as f32;
        let avg_x_weight = x_weights.iter().sum::<f32>() / x_weights.len() as f32;
        let avg_y_weight = y_weights.iter().sum::<f32>() / y_weights.len() as f32;
        let avg_bias_weight = bias_weights.iter().sum::<f32>() / bias_weights.len() as f32;

        println!(
            "weight range: {:?}, avg epochs: {}, x weight: {}, y weight: {}, bias weight: {}",
            weight_range, avg_epochs, avg_x_weight, avg_y_weight, avg_bias_weight
        );
    }
}

fn read_data() -> Vec<DataEntry> {
    let read_string = fs::read_to_string("../data_set.json").unwrap();
    serde_json::from_str(&read_string).unwrap()
}
