import numpy as np
import pickle

from layer import Layer


class MLP:
    def __init__(self, input_number, output_number, hidden_layers, use_bias, learning_rate, momentum,
                 max_epochs, target_error, log_rate, log_path, dataset, save_path):
        # Inicjalizacja podstawowych parametrów sieci
        self.activations = None  # lista aktywacji po każdej warstwie (włącznie z wejściową)
        self.input_number = input_number  # liczba wejść
        self.output_number = output_number  # liczba neuronów wyjściowych
        self.hidden_layers = hidden_layers  # lista liczby neuronów w każdej warstwie ukrytej
        self.use_bias = use_bias  # czy stosować bias
        self.learning_rate = learning_rate  # współczynnik uczenia
        self.momentum = momentum  # współczynnik momentum
        self.max_epochs = max_epochs  # maksymalna liczba epok
        self.target_error = target_error  # docelowy błąd
        self.log_rate = log_rate  # co ile epok logować błąd
        self.log_path = log_path  # ścieżka do pliku logu
        self.dataset = dataset  # dane treningowe (lista: wejście, oczekiwane wyjście)
        self.save_path = save_path  # ścieżka do zapisu wytrenowanej sieci

        self.layers = []  # lista warstw w sieci
        prev_size = input_number  # rozmiar wejścia dla pierwszej warstwy

        for n_neurons in hidden_layers:  # dodanie warstw ukrytych
            self.layers.append(Layer(prev_size, n_neurons, use_bias))  # warstwa z odpowiednią liczbą neuronów
            prev_size = n_neurons  # wyjście tej warstwy staje się wejściem kolejnej

        self.layers.append(Layer(prev_size, output_number, use_bias))  # ostatnia warstwa – wyjściowa

    def forward(self, x):
        # x: wektor wejść numpy array o długości input_number
        # Propagacja sygnału w przód przez wszystkie warstwy
        activations = [x]  # aktywacja warstwy wejściowej to samo wejście
        for layer in self.layers:
            x = layer.forward(x)  # aktywacja kolejnej warstwy
            activations.append(x)  # zapis aktywacji
        self.activations = activations  # Zapis wszystkich aktywacji (np. do logowania)
        return activations[-1]  # zwraca wynik z ostatniej warstwy (wyjście sieci)

    def train(self):
        if self.log_path:
            with open(self.log_path, 'w') as f:
                f.write("epoch,error\n")    # nagłówek pliku logu

        for epoch in range(1, self.max_epochs + 1):  # pętla po epokach
            total_error = 0.0  # suma błędów dla tej epoki
            for x, target in self.dataset:  # pętla po wszystkich wzorcach uczących (para: wejście i oczekiwane wyjście)
                output = self.forward(x)  # propagacja w przód
                delta = output - target  # różnica między odpowiedzią a celem (wektor błędów)

                # propagacja wstecz przez wszystkie warstwy od końca (od warstwy wyjściowej) do początku sieci
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    # wagi kolejnej warstwy (nie ma już „następnej” wtedy W_next = None)
                    W_next = None if i == len(self.layers) - 1 else self.layers[i + 1].W
                    # W przeciwnym razie pobieramy wagi warstwy następnej, potrzebne do obliczenia błędu dla tej warstwy

                    # Przekazujemy błąd do warstwy
                    delta = layer.backward(delta_next=delta, W_next=W_next, learning_rate=self.learning_rate, momentum=self.momentum)

                # Obliczamy średni błąd kwadratowy (MSE) dla danego wzorca i dodajemy do sumy
                total_error += np.mean((target - output) ** 2)  # to służy do monitorowania postępu uczenia się

            avg_error = total_error / len(self.dataset) # średni błąd epoki

            # logowanie
            if epoch % self.log_rate == 0:
                print(f"Epoch {epoch}: error={avg_error:.6f}")
                if self.log_path:
                    with open(self.log_path, 'a') as f:
                        f.write(f"{epoch},{avg_error:.6f}\n")
            # warunek stopu
            if avg_error <= self.target_error:
                print(f"Koniec nauki w epoce {epoch}, error={avg_error:.6f}")
                break

        # opcjonalnie: zapis sieci
        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f)
