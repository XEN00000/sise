import numpy as np
from yt_dlp.utils import lowercase_escape


class Neuron:
    def __init__(self, input_number, activation_func, has_bias):
        self.input_number = input_number
        self.weights = {i: np.random.uniform(-1, 1) for i in range(input_number)}
        self.bias = has_bias
        if has_bias:
            self.bias = np.random.uniform(-1, 1)
        else:
            self.bias = None
        self.activation_func = activation_func.lower()
        self.learning_rate = 0.1

    def calculate_output(self, input_values):
        result = 0
        if self.bias is None:
            for value in range(len(input_values)):
                result += self.weights[value] * input_values[value]
        return result

    def signum_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, input_values):
        if len(input_values) != self.input_number:
            raise Exception("Number of input values does not match the number of neurons")

        z = self.calculate_output(input_values)

        if self.activation_func == "sigmoid":
            return self.signum_activation(z)
        elif self.activation_func == "tanh":
            return np.tanh(z)
        elif self.activation_func == "linear":
            return z
        else:
            raise Exception("Unknown activation function")

