use crate::adaline::Adaline;
use crate::data_entry::DataEntry;
use rand::Rng;
use std::fs;

mod adaline;
mod data_entry;

fn main() {
    let data_entries = read_bipolar_data();

    weight_range_study(&data_entries)
}

/// the first task
fn weight_range_study(data_entries: &Vec<DataEntry>) {
    let learn_factor = 0.01;
    let error_boundary: f32 = 0.21;
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
        let mut reached_errors: Vec<f32> = vec![];
        let mut epochs: Vec<u32> = vec![];

        for _experiment_iter in 0..10 {
            let mut current_epochs = 0;

            let mut adaline = Adaline {
                x_weight: rng.gen_range(weight_range.clone()),
                y_weight: rng.gen_range(weight_range.clone()),
                bias_weight: rng.gen_range(weight_range.clone()),
                delta_square_sum: 0.0,
            };

            loop {
                adaline.delta_square_sum = 0.0;
                current_epochs += 1;

                for entry in data_entries {
                    adaline.learn(entry, learn_factor);
                }

                if (adaline.delta_square_sum / data_entries.len() as f32) < error_boundary {
                    break;
                }
            }

            epochs.push(current_epochs);
            x_weights.push(adaline.x_weight);
            y_weights.push(adaline.y_weight);
            reached_errors.push(adaline.delta_square_sum / data_entries.len() as f32);
            bias_weights.push(adaline.bias_weight);
        }

        let avg_epochs = epochs.iter().sum::<u32>() as f32 / epochs.len() as f32;
        let avg_x_weight = x_weights.iter().sum::<f32>() / x_weights.len() as f32;
        let avg_y_weight = y_weights.iter().sum::<f32>() / y_weights.len() as f32;
        let avg_reached_error = reached_errors.iter().sum::<f32>() / reached_errors.len() as f32;
        let avg_bias_weight = bias_weights.iter().sum::<f32>() / bias_weights.len() as f32;

        println!(
            "weight range: {:?}, avg epochs: {}, reached error: {}, x weight: {}, y weight: {}, bias weight: {}",
            weight_range, avg_epochs, avg_reached_error, avg_x_weight, avg_y_weight, avg_bias_weight
        );
    }
}

fn read_bipolar_data() -> Vec<DataEntry> {
    let read_string = fs::read_to_string("../bipolar_data_set.json").unwrap();
    serde_json::from_str(&read_string).unwrap()
}
