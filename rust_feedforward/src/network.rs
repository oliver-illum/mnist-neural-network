use crate::layer::Layer;
use ndarray::Array2;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, learning_rate: f32) -> Self {
        Self {
            layers,
            learning_rate,
        }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, prediction: &Array2<f32>, target: &Array2<f32>) {
        let mut d_a = crate::loss::cross_entropy_loss_derivative(prediction, target);
        for layer in self.layers.iter_mut().rev() {
            d_a = layer.backward(&d_a, self.learning_rate);
        }
    }
}
