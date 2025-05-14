import numpy as np
import dataLoader as dl
import pickle

from layer import Layer


class MLP:
    def __init__(self, input_number, output_number, hidden_layers, use_bias, learning_rate, momentum,
                 max_epochs, target_error, log_rate, log_path, dataset, save_path):
        self.activations = None
        self.input_number = input_number
        self.output_number = output_number
        self.hidden_layers = hidden_layers
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.target_error = target_error
        self.log_rate = log_rate
        self.log_path = log_path
        self.dataset = dataset
        self.save_path = save_path

        self.layers = []
        prev_size = input_number

        for n_neurons in hidden_layers:
            self.layers.append(Layer(prev_size, n_neurons, use_bias))
            prev_size = n_neurons

        self.layers.append(Layer(prev_size, output_number, use_bias))

    def forward(self, x):
        # x: wektor wejść numpy array o długości input_number
        activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        self.activations = activations
        return activations[-1]

    def train(self):
        """
        Uczy sieć metodą backpropagation z uczącym online.
        """
        for epoch in range(1, self.max_epochs + 1):
            total_error = 0.0
            for x, target in self.dataset:
                # forward
                output = self.forward(x)
                # oblicz błąd i wsteczna propagacja
                delta = output - target
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    # next layer weights, jeśli nie ostatnia warstwa
                    W_next = None if i == len(self.layers) - 1 else self.layers[i + 1].W
                    delta = layer.backward(delta_next=delta,
                                           W_next=W_next,
                                           learning_rate=self.learning_rate,
                                           momentum=self.momentum)
                total_error += np.mean((target - output) ** 2)
            avg_error = total_error / len(self.dataset)

            # logowanie
            if epoch % self.log_rate == 0:
                print(f"Epoch {epoch}: error={avg_error:.6f}")
            # warunek stopu
            if avg_error <= self.target_error:
                print(f"Koniec nauki w epoce {epoch}, error={avg_error:.6f}")
                break

        # opcjonalnie: zapis sieci
        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f)
