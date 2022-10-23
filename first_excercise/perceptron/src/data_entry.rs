use serde::Deserialize;

#[derive(Copy, Clone, Debug, Deserialize)]
pub struct DataEntry {
    pub x: f32,
    pub y: f32,
    pub expected_result: i32,
}
