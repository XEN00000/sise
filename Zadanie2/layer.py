import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, use_bias=True):
        # ustawienia warstwy
        self.use_bias = use_bias
        # wagi i bias
        self.W = np.random.uniform(-0.5, 0.5, (n_neurons, n_inputs))
        self.b = np.random.uniform(-0.5, 0.5, n_neurons) if use_bias else np.zeros(n_neurons)
        # bufory do propagacji i momentum
        self.last_input = None
        self.last_output = None
        self.prev_dW = np.zeros_like(self.W)
        self.prev_db = np.zeros_like(self.b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, y):
        return y * (1 - y)

    def forward(self, x):
        # zapamiętujemy wejście
        self.last_input = x
        z = self.W.dot(x)
        if self.use_bias:
            z += self.b
        # zapisujemy wyjście
        self.last_output = self.sigmoid(z)
        return self.last_output

    def backward(self, delta_next, W_next=None, learning_rate=0.1, momentum=0.0):
        # obliczenie lokalnego błędu
        if W_next is not None:
            error = W_next.T.dot(delta_next)
        else:
            error = delta_next
        delta = error * self.sigmoid_deriv(self.last_output)
        # gradienty wag i biasu
        dW = np.outer(delta, self.last_input)
        db = delta
        # aktualizacja wag z uwzględnieniem momentum
        delta_W = learning_rate * dW + momentum * self.prev_dW
        self.W -= delta_W
        self.prev_dW = delta_W
        if self.use_bias:
            delta_b = learning_rate * db + momentum * self.prev_db
            self.b -= delta_b
            self.prev_db = delta_b
        return delta
