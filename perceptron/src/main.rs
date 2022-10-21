use crate::data_entry::DataEntry;
use perceptron::Perceptron;

use crate::biased_perceptron::BiasedPerceptron;
use rand::Rng;
use std::fs;

mod biased_perceptron;
mod data_entry;
mod perceptron;

fn main() {
    let data_entries = read_data();

    // theta_study(&data_entries);
    // weight_range_study(&data_entries);
    // learn_factor_study(&data_entries);
    bipolar_function_study(&data_entries, &read_bipolar_data());
}

/// the first task
fn theta_study(data_entries: &Vec<DataEntry>) {
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

/// the second task
fn weight_range_study(data_entries: &Vec<DataEntry>) {
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
                bias_weight: rng.gen_range(weight_range.clone()),
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

/// the third task
fn learn_factor_study(data_entries: &Vec<DataEntry>) {
    let mut rng = rand::thread_rng();
    let weight_range = -0.1..0.1;

    for learn_factor in [0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0] {
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
                bias_weight: rng.gen_range(weight_range.clone()),
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
            "learn factor: {}, avg epochs: {}, x weight: {}, y weight: {}, bias weight: {}",
            learn_factor, avg_epochs, avg_x_weight, avg_y_weight, avg_bias_weight
        );
    }
}

// the fourth task
fn bipolar_function_study(data_entries: &Vec<DataEntry>, biased_data_entries: &Vec<DataEntry>) {
    let mut rng = rand::thread_rng();
    let learn_factor = 0.01;
    let weight_range = -0.1..0.1;

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
            bias_weight: rng.gen_range(weight_range.clone()),
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
        "learn factor: {}, avg epochs: {}, x weight: {}, y weight: {}, bias weight: {}",
        learn_factor, avg_epochs, avg_x_weight, avg_y_weight, avg_bias_weight
    );

    let mut bipolar_x_weights: Vec<f32> = vec![];
    let mut bipolar_y_weights: Vec<f32> = vec![];
    let mut bipolar_bias_weights: Vec<f32> = vec![];
    let mut bipolar_epochs: Vec<u32> = vec![];

    for _experiment_iter in 0..10 {
        let mut got_wrong_answer = true;
        let mut current_epochs = 0;

        let mut perceptron = BiasedPerceptron {
            x_weight: rng.gen_range(weight_range.clone()),
            y_weight: rng.gen_range(weight_range.clone()),
            bias_weight: rng.gen_range(weight_range.clone()),
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

        bipolar_epochs.push(current_epochs);
        bipolar_x_weights.push(perceptron.x_weight);
        bipolar_y_weights.push(perceptron.y_weight);
        bipolar_bias_weights.push(perceptron.bias_weight);
    }

    let avg_bipolar_epochs =
        bipolar_epochs.iter().sum::<u32>() as f32 / bipolar_epochs.len() as f32;
    let avg_bipolar_x_weight =
        bipolar_x_weights.iter().sum::<f32>() / bipolar_x_weights.len() as f32;
    let avg_bipolar_y_weight =
        bipolar_y_weights.iter().sum::<f32>() / bipolar_y_weights.len() as f32;
    let avg_bipolar_bias_weight =
        bipolar_bias_weights.iter().sum::<f32>() / bipolar_bias_weights.len() as f32;

    println!(
        "Bipolar: learn factor: {}, avg epochs: {}, x weight: {}, y weight: {}, bias weight: {}",
        learn_factor,
        avg_bipolar_epochs,
        avg_bipolar_x_weight,
        avg_bipolar_y_weight,
        avg_bipolar_bias_weight
    );
}

fn read_data() -> Vec<DataEntry> {
    let read_string = fs::read_to_string("../data_set.json").unwrap();
    serde_json::from_str(&read_string).unwrap()
}

fn read_bipolar_data() -> Vec<DataEntry> {
    let read_string = fs::read_to_string("../bipolar_data_set.json").unwrap();
    serde_json::from_str(&read_string).unwrap()
}
