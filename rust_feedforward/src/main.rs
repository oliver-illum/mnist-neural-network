mod activations;
mod dataset;
mod layer;
mod loss;
mod network;
mod utils;

use dataset::MNISTData;
use layer::{Activation, Layer};
use ndarray::s;
use network::NeuralNetwork;
use utils::*;

fn main() {
    // Load the full MNIST dataset.
    let MNISTData {
        training_images,
        training_labels,
        test_images,
        test_labels,
    } = dataset::load_dataset().unwrap();

    let training_subset = training_images.to_owned();
    let training_labels_subset = training_labels.to_owned();
    let training_labels_onehot = one_hot(&training_labels_subset, 10);

    let layers = vec![
        Layer::new(784, 128, Activation::ReLU),
        Layer::new(128, 10, Activation::Softmax),
    ];
    let mut net = NeuralNetwork::new(layers, 0.01);

    let num_epochs = 101;
    let batch_size = 64;
    let num_samples = training_subset.nrows();

    
    let start_time = std::time::Instant::now();

    
    let initial_predictions = net.forward(&training_subset);
    let initial_loss = loss::cross_entropy_loss(&initial_predictions, &training_labels_onehot);
    let initial_predicted_labels = predictions_to_labels(&initial_predictions);
    let initial_accuracy = compute_accuracy(&initial_predicted_labels, &training_labels_subset);
    println!("Initial Loss = {:.6}, Initial Accuracy = {:.2}%", initial_loss, initial_accuracy * 100.0);

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let batch_images = training_subset
                .slice(s![batch_start..batch_end, ..])
                .to_owned();
            let batch_labels = training_labels_onehot
                .slice(s![batch_start..batch_end, ..])
                .to_owned();

            let predictions = net.forward(&batch_images);
            let loss_value = loss::cross_entropy_loss(&predictions, &batch_labels);
            epoch_loss += loss_value;
            batch_count += 1;

            net.backward(&predictions, &batch_labels);
        }

        epoch_loss /= batch_count as f32;

        
        if epoch % 10 == 0 {
            let final_predictions = net.forward(&training_subset);
            let predicted_labels = predictions_to_labels(&final_predictions);
            let accuracy = compute_accuracy(&predicted_labels, &training_labels_subset);
            println!(
                "Epoch {}: Loss = {:.6}, Training Accuracy = {:.2}%",
                epoch,
                epoch_loss,
                accuracy * 100.0
            );
        }
    }

    
    let final_predictions = net.forward(&training_subset);
    let predicted_labels = predictions_to_labels(&final_predictions);
    let final_accuracy = compute_accuracy(&predicted_labels, &training_labels_subset);
    println!("\nFinal Training Accuracy = {:.2}%", final_accuracy * 100.0);

    let test_predictions = net.forward(&test_images);
    let test_predicted_labels = predictions_to_labels(&test_predictions);
    let test_accuracy = compute_accuracy(&test_predicted_labels, &test_labels);

    println!("Test Accuracy = {:.2}%\n", test_accuracy * 100.0);

    println!("Sample test predictions:");
    for i in 0..10 {
        println!(
            "Sample {}: Predicted = {}, Actual = {}",
            i, test_predicted_labels[i], test_labels[i]
        );
    }

    
    let duration = start_time.elapsed();
    let total_seconds = duration.as_secs_f64();
    let minutes = total_seconds as u64 / 60;
    let seconds = total_seconds % 60.0;
    println!("\nTotal execution time: {} minutes and {:.2} seconds", minutes, seconds);
}

#[allow(unused)]
fn print_dataset() {
    let MNISTData {
        training_images,
        training_labels,
        test_images,
        test_labels,
    } = dataset::load_dataset().unwrap();

    println!(
        "First training image (flattened):\n{:?}",
        training_images.row(0)
    );
    println!("First training label: {:?}", training_labels[0]);

    println!("\nFirst five training images:");
    for i in 0..5 {
        println!("Image {}:\n{:?}", i, training_images.row(i));
    }

    println!("\nFirst five training labels:");
    for i in 0..5 {
        println!("Label {}: {:?}", i, training_labels[i]);
    }

    println!("\nFirst test image (flattened):\n{:?}", test_images.row(0));
    println!("First test label: {:?}", test_labels[0]);

    println!("\nFirst five test images:");
    for i in 0..5 {
        println!("Test Image {}:\n{:?}", i, test_images.row(i));
    }

    println!("\nFirst five test labels:");
    for i in 0..5 {
        println!("Test Label {}: {:?}", i, test_labels[i]);
    }
}
