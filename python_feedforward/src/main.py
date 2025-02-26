import numpy as np

from dataset import load_dataset
from layer import Layer
from network import NeuralNetwork
from loss import cross_entropy_loss
from utils import one_hot, predictions_to_labels, compute_accuracy

def run_main():
    train_images, train_labels, test_images, test_labels = load_dataset()

    num_train = train_images.shape[0]
    num_test = test_images.shape[0]
    X_train = train_images.reshape(num_train, -1).astype(np.float32) / 255.0
    X_test = test_images.reshape(num_test, -1).astype(np.float32) / 255.0

    y_train = one_hot(train_labels, 10)
    y_test = one_hot(test_labels, 10)

    layer1 = Layer(input_dim=X_train.shape[1], output_dim=128, activation='relu')
    layer2 = Layer(input_dim=128, output_dim=10, activation='softmax')
    network = NeuralNetwork(layers=[layer1, layer2], learning_rate=0.01)

    epochs = 100
    batch_size = 64
    
    initial_predictions = network.forward(X_train)
    initial_loss = cross_entropy_loss(initial_predictions, y_train)
    initial_train_predictions = predictions_to_labels(initial_predictions)
    initial_train_actual = np.argmax(y_train, axis=1)
    initial_accuracy = compute_accuracy(initial_train_predictions, initial_train_actual)
    print(f"Initial Loss: {initial_loss:.4f}, Initial Train Acc: {initial_accuracy:.4f}")

    for epoch in range(epochs):
        permutation = np.random.permutation(num_train)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        num_batches = int(np.ceil(num_train / batch_size))
        epoch_loss = 0.0

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_train)
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            predictions = network.forward(X_batch)
            loss = cross_entropy_loss(predictions, y_batch)
            epoch_loss += loss * (end - start) 

            network.backward(predictions, y_batch)

        epoch_loss /= num_train

        train_predictions = predictions_to_labels(network.forward(X_train))
        train_actual = np.argmax(y_train, axis=1)
        train_accuracy = compute_accuracy(train_predictions, train_actual)

        test_predictions = predictions_to_labels(network.forward(X_test))
        test_actual = np.argmax(y_test, axis=1)
        test_accuracy = compute_accuracy(test_predictions, test_actual)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        
    final_test_predictions = predictions_to_labels(network.forward(X_test))
    final_test_actual = np.argmax(y_test, axis=1)
    print("\nSome test predictions:")
    for i in range(10):
        print(f"Test Image {i}: Predicted: {final_test_predictions[i]}, Actual: {final_test_actual[i]}")

if __name__ == '__main__':
    run_main()
