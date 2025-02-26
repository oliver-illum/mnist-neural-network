use ndarray::{Array1, Array2};

pub fn one_hot(labels: &Array1<u8>, num_classes: usize) -> Array2<f32> {
    let num_labels = labels.len();
    let mut one_hot = Array2::<f32>::zeros((num_labels, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        one_hot[[i, label as usize]] = 1.0;
    }
    one_hot
}

// Compute the predicted class for each example using argmax.
pub fn predictions_to_labels(predictions: &Array2<f32>) -> Vec<u8> {
    predictions
        .outer_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u8)
                .unwrap_or(0)
        })
        .collect()
}

pub fn compute_accuracy(predicted: &[u8], actual: &Array1<u8>) -> f32 {
    let correct = predicted
        .iter()
        .zip(actual.iter())
        .filter(|(p, a)| p == a)
        .count();
    correct as f32 / predicted.len() as f32
}
