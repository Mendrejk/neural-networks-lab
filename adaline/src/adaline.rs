use crate::data_entry::DataEntry;

pub struct Adaline {
    pub x_weight: f32,
    pub y_weight: f32,
    pub bias_weight: f32,
    pub delta_square_sum: f32,
}

impl Adaline {
    pub fn learn(&mut self, data_entry: &DataEntry, learn_factor: f32) {
        let stimulus =
            data_entry.x * self.x_weight + data_entry.y * self.y_weight + self.bias_weight;

        let delta = data_entry.expected_result as f32 - stimulus;
        self.delta_square_sum += delta * delta;

        self.x_weight += learn_factor * delta * data_entry.x;
        self.y_weight += learn_factor * delta * data_entry.y;
        self.bias_weight += learn_factor * delta;
    }

    pub fn check(&self, data_entry: &DataEntry) -> bool {
        let stimulus =
            data_entry.x * self.x_weight + data_entry.y * self.y_weight + self.bias_weight;

        let result = if stimulus > 0.0 { 1 } else { -1 };

        data_entry.expected_result - result == 0
    }
}
