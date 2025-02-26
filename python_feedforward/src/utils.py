import numpy as np

def one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot_encoded = np.zeros((num_labels, num_classes), dtype=np.float32)
    one_hot_encoded[np.arange(num_labels), labels] = 1.0
    return one_hot_encoded

def predictions_to_labels(predictions):
    return np.argmax(predictions, axis=1)

def compute_accuracy(predicted, actual):
    correct = np.sum(predicted == actual)
    return correct / len(predicted)
