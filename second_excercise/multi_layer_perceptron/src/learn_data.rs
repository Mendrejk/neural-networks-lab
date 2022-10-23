use ndarray::Array1;

pub struct LearnData {
    image_parts: Vec<f64>, // TODO - probably vec of vectors? or sth
    expected_class: f64,   // TODO - enum
}

impl LearnData {
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from(self.image_parts.clone())
    }
}
