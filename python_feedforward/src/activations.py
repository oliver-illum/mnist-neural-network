import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    """
    When combining softmax with cross-entropy loss, the derivative simplifies.
    Here we return an array of ones to mimic the Rust code behavior.
    """
    return np.ones_like(x)