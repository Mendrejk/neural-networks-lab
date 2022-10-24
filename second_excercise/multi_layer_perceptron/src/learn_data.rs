use ndarray::Array1;

pub struct LearnData {
    pub(crate) image_parts: Vec<f64>,
    // TODO - probably vec of vectors? or sth
    pub expected_class: usize, // TODO - enum
}

impl LearnData {
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from(self.image_parts.clone())
    }
}
