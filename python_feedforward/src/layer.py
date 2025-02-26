import numpy as np
from activations import *

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32)
        self.biases = np.zeros((1, output_dim), dtype=np.float32)
        self.activation = activation.lower()
        self.last_input = None
        self.last_z = None

    def forward(self, input):
        self.last_input = input
        self.last_z = np.dot(input, self.weights) + self.biases
        
        if self.activation == 'relu':
            return relu(self.last_z)
        elif self.activation == 'softmax':
            return softmax(self.last_z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def backward(self, d_a, learning_rate):
        if self.activation == 'relu':
            d_activation = relu_derivative(self.last_z)
        elif self.activation == 'softmax':
            d_activation = softmax_derivative(self.last_z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
        
        d_z = d_a * d_activation
        
        d_w = np.dot(self.last_input.T, d_z)
        d_b = np.sum(d_z, axis=0, keepdims=True)
        
        d_x = np.dot(d_z, self.weights.T)
        
        self.weights -= learning_rate * d_w
        self.biases -= learning_rate * d_b
        
        return d_x
