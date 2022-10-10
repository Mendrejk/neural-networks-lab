use serde::Serialize;

#[derive(Copy, Clone, Debug, Serialize)]
pub struct DataEntry {
    pub x: f32,
    pub y: f32,
    pub expected_result: i32,
}
