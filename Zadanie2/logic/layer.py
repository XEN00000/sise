import numpy as np


class Layer:
    def __init__(self, input_number, neuron_number):
        self.output = None
        self.weights = np.random.randn(input_number, neuron_number)
        self.biases = np.zeros((1, neuron_number))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases