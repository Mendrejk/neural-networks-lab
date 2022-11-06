use crate::config::{IMAGE_DIMENSION, IMAGE_SIZE};
use mnist::MnistBuilder;
use ndarray::{Array1, Array2};

pub struct LearnData {
    pub image_parts: Array2<u8>,
    pub expected_class: u8, // TODO - enum
}

impl LearnData {
    pub fn to_neural_input(&self) -> Array1<f64> {
        Array1::from_iter(self.image_parts.iter().cloned().map(|part| part as f64))
    }
}

impl LearnData {
    pub fn load_mnist() -> (Vec<Self>, Vec<Self>) {
        let mnist = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(60_000)
            .test_set_length(10_000)
            .finalize();

        let train_data = mnist
            .trn_img
            .chunks(IMAGE_SIZE)
            .map(|chunk| Array2::from_shape_vec((IMAGE_DIMENSION, IMAGE_DIMENSION), chunk.into()))
            .zip(mnist.trn_lbl.into_iter())
            .map(|(image_parts, expected_class)| Self {
                image_parts: image_parts.unwrap(),
                expected_class,
            })
            .collect();
        let test_data = mnist
            .tst_img
            .chunks(IMAGE_SIZE)
            .map(|chunk| Array2::from_shape_vec((IMAGE_DIMENSION, IMAGE_DIMENSION), chunk.into()))
            .zip(mnist.tst_lbl.into_iter())
            .map(|(image_parts, expected_class)| Self {
                image_parts: image_parts.unwrap(),
                expected_class,
            })
            .collect();

        (train_data, test_data)
    }
}
