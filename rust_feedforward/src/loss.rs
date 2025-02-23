#![allow(unused)]
use ndarray::{Array2, Axis};

pub fn cross_entropy_loss(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
    // small epsilon to prevent log(0)
    let epsilon = 1e-8;
    let clipped_predictions = predictions.mapv(|x| x.max(epsilon));
    let log_preds = clipped_predictions.mapv(|x| x.ln());
    let loss_matrix = targets * log_preds;
    let loss_per_example = loss_matrix.sum_axis(Axis(1)).mapv(|sum| -sum);
    loss_per_example.mean().unwrap()
}

// When using softmax followed by cross-entropy, this derivative
// simplifies to: predictions - targets.
pub fn cross_entropy_loss_derivative(
    predictions: &Array2<f32>,
    targets: &Array2<f32>,
) -> Array2<f32> {
    predictions - targets
}
