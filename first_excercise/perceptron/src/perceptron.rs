use crate::DataEntry;

pub struct Perceptron {
    pub x_weight: f32,
    pub y_weight: f32,
}

impl Perceptron {
    pub fn learn(&mut self, data_entry: &DataEntry, learn_factor: f32, theta: f32) -> bool {
        let stimulus = data_entry.x * self.x_weight + data_entry.y * self.y_weight;

        let result = if stimulus > theta { 1 } else { 0 };
        let delta = (data_entry.expected_result - result) as f32;

        self.x_weight += learn_factor * delta * data_entry.x;
        self.y_weight += learn_factor * delta * data_entry.y;

        delta == 0.0
    }
}
