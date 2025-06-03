import pickle
from tkinter import Button
import tkinter as tk

import numpy as np
from matplotlib import pyplot as plt

import dataLoader
from mlp import MLP


class IrisNetwork:
    def __init__(self, root):
        self.entry_use_bias = None
        self.entry_is_shuffled = None
        self.root = root
        self.root.title("MLP - Iris set")
        self.root.geometry("600x700")

        # 1. Zbiór treningowy i testowy
        self.training_dataset, self.testing_dataset = None, None

        self.errors = []

        fields = [
            ("Input neurons", "entry_input", 4),
            ("Output neurons", "entry_output", 3),
            ("Hidden layers (comma-separated)", "entry_hidden", "5,4"),
            ("Learning rate", "entry_lr", 0.01),
            ("Momentum", "entry_momentum", 0.9),
            ("Epochs", "entry_epochs", 500),
            ("Target error", "entry_error", 1e-4),
            ("Log every N epochs", "entry_log_rate", 50),
            ("Log file path", "entry_log_path",
             "C:/Users/lukas/Documents/Moje/Studia/4_semestr/sise/Zadanie2/iris_log.csv"),
            ("Data set", "entry_data_set",
             "C:/Users/lukas/Documents/Moje/Studia/4_semestr/sise/Zadanie2/irisset.csv"),
            ("% size of training set (0-1 value)", "entry_training_size", 0.7),
            ("Network save path", "entry_save_path",
             "C:/Users/lukas/Documents/Moje/Studia/4_semestr/sise/Zadanie2/irisNetwork.ntwrk"),
        ]

        for label, attr, cont in fields:
            tk.Label(self.root, text=label).pack()

            # 1. Tworzymy jedną instancję Entry(...)
            entry_widget = tk.Entry(self.root)

            # 2. Wstawiamy wartość domyślną
            entry_widget.insert(0, str(cont))

            # 3. Przypisujemy widget do self.xxx
            setattr(self, attr, entry_widget)

            # 4. Pakuje widget
            entry_widget.pack()

        bias_checkbox = tk.Checkbutton(self.root, text="Bias", variable=self.entry_use_bias, onvalue=True,
                                       offvalue=False)
        bias_checkbox.pack()
        shuffled_checkbox = tk.Checkbutton(self.root, text="Shuffle data set", variable=self.entry_is_shuffled,
                                           onvalue=True,
                                           offvalue=False)
        shuffled_checkbox.pack()
        setup_button = Button(self.root, text="Setup", command=self.net_setup)
        setup_button.pack()
        train_button = Button(self.root, text="Train", command=self.train)
        train_button.pack()
        benchmark_button = Button(self.root, text="Test", command=self.benchmark)
        benchmark_button.pack()

    def net_setup(self):
        self.input_num = int(self.entry_input.get())
        self.output_num = int(self.entry_output.get())
        self.hidden_layers = list(map(int, self.entry_hidden.get().split(',')))
        self.learning_rate = float(self.entry_lr.get())
        self.momentum = float(self.entry_momentum.get())
        self.epochs = int(self.entry_epochs.get())
        self.error = float(self.entry_error.get())
        self.use_bias = bool(self.entry_use_bias)
        self.log_rate = int(self.entry_log_rate.get())
        self.log_path = str(self.entry_log_path.get())
        self.data_set = str(self.entry_data_set.get())
        self.training_size = float(self.entry_training_size.get())
        self.is_shuffled = bool(self.entry_is_shuffled)
        self.save_path = str(self.entry_save_path.get())
        #self.training_dataset, self.testing_dataset = dataLoader.loader(dataset_name=self.data_set, shuffled=self.is_shuffled, training_set_size=self.training_size)
        self.training_dataset, self.testing_dataset = dataLoader.loader(dataset_name=self.data_set)

    def train(self):
        try:
            net = MLP(input_number=self.input_num,
                      output_number=self.output_num,
                      hidden_layers=self.hidden_layers,
                      use_bias=self.use_bias,
                      learning_rate=self.learning_rate,
                      momentum=self.momentum,
                      max_epochs=self.epochs,
                      target_error=self.error,
                      log_rate=self.log_rate,
                      log_path=self.log_path,
                      dataset=self.training_dataset,
                      save_path=self.save_path)

            net.train()
            net.save()
            errors = net.error_history
            epochs = [i * self.log_rate for i in range(1, len(errors) + 1)]
            self.draw_learning_plot(errors, epochs)

        except ValueError as ve:
            print(f"Invalid entry: {ve}")

    def benchmark(self):
        self.test_the_network(self.testing_dataset, self.save_path, "C:/Users/lukas/Documents/Moje/Studia/4_semestr/sise/Zadanie2/iris_test_log.csv")

    def draw_learning_plot(self, errors, epochs):
        # Rysujemy wykres
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, errors, label="MSE każdej epoki")
        plt.xlabel("Epoka")
        plt.ylabel("Średni błąd (MSE)")
        plt.title("Krzywa błędu w trakcie uczenia")
        plt.grid(True)
        plt.legend()
        plt.show()

    # Funkcja testująca wytrenowaną sieć na zadanym zbiorze danych
    def test_the_network(self, dataset, network_path='irisNet.ntwrk', log_path=None):
        # Wczytanie wytrenowanej sieci z pliku
        with open(network_path, 'rb') as f:
            loaded_net = pickle.load(f)  # załadowanie sieci z pliku

        # Przygotowanie pliku logu (jeśli podano)
        f_log = open(log_path, 'w') if log_path else None
        if f_log:
            f_log.write("input;target;prediction;error;hidden_output;output_weights\n")

        y_true = []
        y_pred = []

        for x, y in dataset:
            out = loaded_net.forward(x)
            pred = np.argmax(out)
            true = np.argmax(y)

            y_pred.append(pred)
            y_true.append(true)

            error_vec = y - out
            hidden_output = loaded_net.activations[1]
            output_weights = loaded_net.layers[-1].W

            if f_log:
                f_log.write(f"{x.tolist()};{y.tolist()};{out.tolist()};"
                            f"{np.mean(np.abs(error_vec)):.4f};"
                            f"{hidden_output.tolist()};"
                            f"{output_weights.tolist()}\n")

        if f_log:
            f_log.close()

        # Macierz pomyłek
        n_classes = loaded_net.output_number
        conf_mat = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            conf_mat[t, p] += 1

        accuracy = np.trace(conf_mat) / conf_mat.sum() if conf_mat.sum() != 0 else 0

        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.diag(conf_mat) / conf_mat.sum(axis=0)
            recall = np.diag(conf_mat) / conf_mat.sum(axis=1)

        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)

        f1 = np.zeros_like(precision)
        for i in range(len(precision)):
            if precision[i] + recall[i] == 0:
                f1[i] = 0.0
            else:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        # Wyświetlenie wyników
        print(f'\nDokładność testu: {accuracy:.4f}\n')
        print('Macierz pomyłek:')
        print(conf_mat, '\n')

        for i in range(n_classes):
            print(f'Klasa {i}: Precyzja={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}')
        if log_path:
            f_log.close()


if __name__ == "__main__":
    root_window = tk.Tk()
    app = IrisNetwork(root_window)
    root_window.mainloop()
