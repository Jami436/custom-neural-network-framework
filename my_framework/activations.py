import numpy as np
from .base import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.copy()
        input_gradient[self.input <= 0] = 0
        return input_gradient
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        clipped_input = np.clip(input_data, -500, 500)
        self.output = 1/(1 + np.exp(-clipped_input))
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        sigmoid_derivative = self.output * (1 - self.output)
        input_gradient = output_gradient * sigmoid_derivative
        return input_gradient