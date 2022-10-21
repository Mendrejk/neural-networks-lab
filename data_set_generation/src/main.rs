use crate::data_entry::DataEntry;

use std::fs;
use std::fs::File;

use rand::Rng;
use rand::seq::SliceRandom;

mod data_entry;

fn main() {
    serialise_bipolar_data_set()
}

fn serialise_data_set() {
    let data_set = generate_data_set();

    // serialise
    let serialised = serde_json::to_string(&data_set).unwrap();

    File::create("../data_set.json").unwrap();
    fs::write("../data_set.json", serialised).unwrap();
}

fn serialise_bipolar_data_set() {
    let data_set = generate_bipolar_data_set();

    // serialise
    let serialised = serde_json::to_string(&data_set).unwrap();

    File::create("../bipolar_data_set.json").unwrap();
    fs::write("../bipolar_data_set.json", serialised).unwrap();
}

fn generate_data_set() -> Vec<DataEntry> {
    let negative_data_entries = vec![
        DataEntry {
            x: 0.0,
            y: 0.0,
            expected_result: 0,
        },
        DataEntry {
            x: 1.0,
            y: 0.0,
            expected_result: 0,
        },
        DataEntry {
            x: 0.0,
            y: 1.0,
            expected_result: 0,
        },
    ];
    let positive_data_entries = vec![DataEntry {
        x: 1.0,
        y: 1.0,
        expected_result: 1,
    }];

    let mut generated_entries: Vec<DataEntry> = vec![];
    generated_entries.append(&mut negative_data_entries.clone());
    generated_entries.append(&mut positive_data_entries.clone());

    let mut rng = rand::thread_rng();

    for _entry_index in generated_entries.len()..=35 {
        let is_positive = rng.gen::<bool>();

        let origin_entry: &DataEntry = (if is_positive {
            &positive_data_entries
        } else {
            &negative_data_entries
        })
            .choose(&mut rng)
            .unwrap();

        let expected_result = if is_positive { 1 } else { 0 };
        generated_entries.push(DataEntry {
            x: rng.gen_range((origin_entry.x - 0.1)..(origin_entry.x + 0.1)),
            y: rng.gen_range((origin_entry.y - 0.1)..(origin_entry.y + 0.1)),
            expected_result,
        });
    }

    generated_entries
}

fn generate_bipolar_data_set() -> Vec<DataEntry> {
    let negative_data_entries = vec![
        DataEntry {
            x: -1.0,
            y: -1.0,
            expected_result: -1,
        },
        DataEntry {
            x: 1.0,
            y: -1.0,
            expected_result: -1,
        },
        DataEntry {
            x: -1.0,
            y: 1.0,
            expected_result: -1,
        },
    ];
    let positive_data_entries = vec![DataEntry {
        x: 1.0,
        y: 1.0,
        expected_result: 1,
    }];

    let mut generated_entries: Vec<DataEntry> = vec![];
    generated_entries.append(&mut negative_data_entries.clone());
    generated_entries.append(&mut positive_data_entries.clone());

    let mut rng = rand::thread_rng();

    for _entry_index in generated_entries.len()..=35 {
        let is_positive = rng.gen::<bool>();

        let origin_entry: &DataEntry = (if is_positive {
            &positive_data_entries
        } else {
            &negative_data_entries
        })
            .choose(&mut rng)
            .unwrap();

        let expected_result = if is_positive { 1 } else { -1 };
        generated_entries.push(DataEntry {
            x: rng.gen_range((origin_entry.x - 0.1)..(origin_entry.x + 0.1)),
            y: rng.gen_range((origin_entry.y - 0.1)..(origin_entry.y + 0.1)),
            expected_result,
        });
    }

    generated_entries
}
