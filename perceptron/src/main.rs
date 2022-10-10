use crate::data_entry::DataEntry;
use perceptron::Perceptron;

use std::fs;

mod data_entry;
mod perceptron;
mod biased_perceptron;

fn main() {
    let data_entries = read_data();

    first_task(&data_entries);
}

fn first_task(data_entries: &Vec<DataEntry>) {
    let learn_factor = 0.1;
    let mut perceptron = Perceptron {
        x_weight: 0.01,
        y_weight: 0.01,
    };

    for theta in [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2] {
        let mut x_weights: Vec<f32> = vec![];
        let mut y_weights: Vec<f32> = vec![];
        let mut epochs: Vec<u32> = vec![];

        let mut current_epochs = 0;
        let mut got_wrong_answer = true;

        for _experiment_iter in 0..10 {
            while got_wrong_answer {
                got_wrong_answer = false;

                for entry in &data_entries {
                    if !perceptron.learn(entry, learn_factor, theta) {
                        got_wrong_answer = true;
                    }
                }

                current_epochs += 1;
            }
        }

        println!("epochs: {}", epochs); // TODO remove those 3
        println!("x weights: {}", x_weights);
        println!("y weights: {}", y_weights);

        let avg_epochs = epochs.iter().sum::<u32>() as f32 / epochs.len() as f32;
        let avg_x_weight = x_weights.iter().sum::<f32>() / x_weight.len() as f32;
        let avg_y_weight = y_weights.iter().sum::<f32>() / y_weight.len() as f32;
        println!("theta: {}, avg epochs: {}, x weight: {}, y weight: {}", theta, avg_epochs, avg_x_weight, avg_y_weight);
    }
}

fn read_data() -> Vec<DataEntry> {
    let read_string = fs::read_to_string("../data_set.json").unwrap();
    serde_json::from_str(&read_string).unwrap()
}
