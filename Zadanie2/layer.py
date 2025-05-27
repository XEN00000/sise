import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons, use_bias=True):
        # ustawienia warstwy
        self.use_bias = use_bias
        # wagi i bias
        self.W = np.random.uniform(-0.5, 0.5, (n_neurons, n_inputs))  # inicjalizacja wag losowymi wartościami
        self.b = np.random.uniform(-0.5, 0.5, n_neurons) if use_bias else np.zeros(n_neurons)  # inicjalizacja biasu
        # bufory do propagacji i momentum
        self.last_input = None  # bufor ostatniego wejścia
        self.last_output = None  # bufor ostatniego wyjścia
        self.prev_dW = np.zeros_like(self.W)  # bufor poprzednich zmian wag (dla momentum)
        self.prev_db = np.zeros_like(self.b)  # bufor poprzednich zmian biasu (dla momentum)

    def sigmoid(self, x):
        # zwraca wynik sigmoidy
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, y):
        # klasyczna pochodna sigmoidy
        return y * (1 - y)

    def forward(self, x):
        # zapamiętujemy wejście
        self.last_input = x
        z = self.W.dot(x)  # iloczyn wag i wejścia
        if self.use_bias:  # dodanie biasu (jeśli używany)
            z += self.b
        # zapisujemy wyjście
        self.last_output = self.sigmoid(z)  # zastosowanie funkcji aktywacji i zapamiętanie wyjścia
        return self.last_output  # zwrócenie aktywacji warstwy

    def backward(self, delta_next, W_next=None, learning_rate=0.1, momentum=0.0):
        # obliczenie lokalnego błędu
        # W_next - macierz wag z warstwy następnej
        if W_next is not None:  # jeśli nie jesteśmy w ostatniej warstwie
            error = W_next.T.dot(delta_next)  # przeliczenie błędu na podstawie wag kolejnej warstwy
            # Przenosi błąd z warstwy następnej do bieżącej (sygnał „wstecz”)
        else:
            error = delta_next  # w ostatniej warstwie przekazujemy błąd bezpośrednio

        # delta: błąd lokalny * pochodna sigmoidy
        delta = error * self.sigmoid_deriv(self.last_output)

        # Gradient wag: iloczyn zewnętrzny (outer product) delta i wejścia
        dW = np.outer(delta, self.last_input)  # tworzy macierz gradientów wag dla każdego połączenia wejście->neuron
        # Tworzy pełną macierz zmian wag: jak bardzo każdy neuron powinien zmienić wagę względem każdego wejścia

        # Gradient biasu: to samo co delta (bo bias nie zależy od wejścia)
        db = delta

        # Obliczanie zmiany wag z uwzględnieniem momentum
        delta_W = learning_rate * dW + momentum * self.prev_dW  # aktualizacja wag z momentum
        self.W -= delta_W  # zastosowanie zmiany wag
        self.prev_dW = delta_W  # zapisanie ostatniej zmiany wag

        # Aktualizacja biasów
        if self.use_bias:  # jeśli bias aktywny
            delta_b = learning_rate * db + momentum * self.prev_db  # aktualizacja biasu z momentum
            self.b -= delta_b  # zastosowanie zmiany biasu
            self.prev_db = delta_b  # zapisanie ostatniej zmiany biasu

        return delta  # zwrócenie błędu do poprzedniej warstwy
