#![allow(unused)]
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
// use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::activations;

#[derive(Clone, Debug)]
pub enum Activation {
    ReLU,
    Softmax,
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: Activation,

    pub last_input: Option<Array2<f32>>,
    pub last_z: Option<Array2<f32>>,
}

impl Layer {
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        // using a normal distribution.
        let weights = Array2::<f32>::random((input_dim, output_dim), StandardNormal);
        let biases = Array1::<f32>::zeros(output_dim);
        Self {
            weights,
            biases,
            activation,
            last_input: None,
            last_z: None,
        }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());
        let z = input.dot(&self.weights) + &self.biases;
        self.last_z = Some(z.clone());
        crate::activations::apply(&self.activation, &z)
    }

    pub fn backward(&mut self, d_a: &Array2<f32>, lr: f32) -> Array2<f32> {
        let z = self.last_z.as_ref().expect("Cached z not found");
        let input = self.last_input.as_ref().expect("Cached input not found");

        // derivative of the activation function.
        let d_activation = activations::derivative(&self.activation, z);

        // combine with the gradient coming from the next layer.
        let d_z = d_a * &d_activation;

        // gradients with respect to weights and biases.
        let d_w = input.t().dot(&d_z);
        let db = d_z.sum_axis(Axis(0));

        // the gradient to the previous layer.
        let d_x = d_z.dot(&self.weights.t());

        self.weights -= &(lr * &d_w);
        self.biases -= &(lr * &db);

        d_x
    }
}
