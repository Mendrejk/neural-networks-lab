pub static STANDARD_DISTRIBUTION: f32 = 1.0;
pub static OUTPUT_SIZE: usize = 10;
pub static IMAGE_DIMENSION: usize = 28;
pub static IMAGE_SIZE: usize = IMAGE_DIMENSION * IMAGE_DIMENSION;
pub static BATCH_SIZE: usize = 50;
pub static LEARN_FACTOR: f64 = 0.1;
pub static UPDATE_FACTOR: f64 = LEARN_FACTOR / BATCH_SIZE as f64;
