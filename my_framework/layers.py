import numpy as np
from .base import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) * np.sqrt(2.0/input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward(self, output_gradient, optimizer):
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0).reshape(1, -1)

        input_gradient = np.dot(output_gradient, self.weights.T)

        optimizer.update(self, weights_gradient, biases_gradient)

        return input_gradient