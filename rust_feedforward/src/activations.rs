use crate::layer::Activation;
use ndarray::Array2;

pub fn apply(activation: &Activation, x: &Array2<f32>) -> Array2<f32> {
    match activation {
        Activation::ReLU => relu(x),
        Activation::Softmax => softmax(x),
    }
}

fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| if v > 0.0 { v } else { 0.0 })
}

// fn softmax(x: &Array2<f32>) -> Array2<f32> {
//     let exp_x = x.mapv(|v| v.exp());
//     let sum_exp = exp_x
//         .sum_axis(ndarray::Axis(1))
//         .insert_axis(ndarray::Axis(1));
//     exp_x / sum_exp
// }

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let max_per_row = x.map_axis(ndarray::Axis(1), |row| {
        row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    });
    let max_per_row = max_per_row.insert_axis(ndarray::Axis(1));
    let exp_x = (x - &max_per_row).mapv(|v| v.exp());
    let sum_exp = exp_x
        .sum_axis(ndarray::Axis(1))
        .insert_axis(ndarray::Axis(1));
    exp_x / sum_exp
}

pub fn derivative(activation: &Activation, x: &Array2<f32>) -> Array2<f32> {
    match activation {
        Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
        Activation::Softmax => Array2::ones(x.raw_dim()),
    }
}
