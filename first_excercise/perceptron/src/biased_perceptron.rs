use crate::DataEntry;

pub struct BiasedPerceptron {
    pub x_weight: f32,
    pub y_weight: f32,
    pub bias_weight: f32,
}

impl BiasedPerceptron {
    pub fn learn(&mut self, data_entry: &DataEntry, learn_factor: f32) -> bool {
        let stimulus =
            data_entry.x * self.x_weight + data_entry.y * self.y_weight + self.bias_weight;

        let result = if stimulus > 0.0 { 1 } else { 0 };
        let delta = (data_entry.expected_result - result) as f32;

        self.x_weight += learn_factor * delta * data_entry.x;
        self.y_weight += learn_factor * delta * data_entry.y;
        self.bias_weight += learn_factor * delta;

        delta == 0.0
    }
}
