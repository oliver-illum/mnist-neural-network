from loss import cross_entropy_loss_derivative

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, prediction, target):
        d_a = cross_entropy_loss_derivative(prediction, target)
        for layer in reversed(self.layers):
            d_a = layer.backward(d_a, self.learning_rate)
