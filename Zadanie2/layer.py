import random
from acivations import *

class Layer:
    def __init__(self, n_inputs, n_neurons, use_bias=True):
        self.use_bias = use_bias
        self.weights = [
            [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
            for _ in range(n_neurons)
        ]
        # jeśli bias wyłączony, po prostu zero
        self.biases = (
            [random.uniform(-0.5, 0.5) for _ in range(n_neurons)]
            if use_bias else
            [0.0] * n_neurons
        )

    def forward(self, inputs):
        self.inputs = inputs[:]
        self.net = []
        self.outputs = []
        for w_row, b in zip(self.weights, self.biases):
            net_i = sum(i * w for i, w in zip(self.inputs, w_row))
            if self.use_bias:
                net_i += b
            out_i = sigmoid(net_i)
            self.net.append(net_i)
            self.outputs.append(out_i)
        return self.outputs

    def backward(self, output_errors, learning_rate, momentum, prev_weight_updates):
        deltas = [
            err * sigmoid_derivative(out)
            for err, out in zip(output_errors, self.outputs)
        ]
        prev_errors = [0.0] * len(self.inputs)
        for i, w_row in enumerate(self.weights):
            for j, w_ij in enumerate(w_row):
                prev_errors[j] += w_ij * deltas[i]

        weight_updates = [
            [0.0] * len(self.inputs)
            for _ in range(len(self.weights))
        ]
        for i, w_row in enumerate(self.weights):
            for j in range(len(self.inputs)):
                update = learning_rate * deltas[i] * self.inputs[j]
                if momentum:
                    update += momentum * prev_weight_updates[i][j]
                self.weights[i][j] += update
                weight_updates[i][j] = update
            if self.use_bias:
                self.biases[i] += learning_rate * deltas[i]

        return prev_errors, weight_updates