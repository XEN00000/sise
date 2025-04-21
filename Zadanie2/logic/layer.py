import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases