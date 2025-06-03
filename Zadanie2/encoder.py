from tkinter import Button
from tkinter import messagebox

import numpy as np
import tkinter as tk

from matplotlib import pyplot as plt

from mlp import MLP
from plotter import draw_learning_plot


class Encoder:
    def __init__(self, root):
        self.entry_use_bias = tk.BooleanVar()
        self.root = root
        self.root.title("MLP - Encoder")
        self.root.geometry("600x600")

        # 1. Wzorce auto enkodera
        self.patterns = [
            (np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])),
            (np.array([0, 1, 0, 0]), np.array([0, 1, 0, 0])),
            (np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0])),
            (np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1])),
        ]

        self.errors = []

        fields = [
            ("Input neurons", "entry_input", 4),
            ("Output neurons", "entry_output", 4),
            ("Hidden layers (comma-separated)", "entry_hidden", 2),
            ("Learning rate", "entry_lr", 0.6),
            ("Momentum", "entry_momentum", 0),
            ("Epochs", "entry_epochs", 1000),
            ("Target error", "entry_error", 1e-4),
            ("Log every N epochs", "entry_log_rate", 100),
            ("Log file path", "entry_log_path",
             "encoder_log.txt"),
            ("Network save path", "entry_save_path",
             "encoder.ntwrk"),
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
        setup_button = Button(self.root, text="Setup", command=self.net_setup)
        setup_button.pack()
        train_button = Button(self.root, text="Train", command=self.experiment)
        train_button.pack()
        benchmark_button = Button(self.root, text="Benchmark", command=self.benchmark)
        benchmark_button.pack()

    def net_setup(self):
        self.input_num = int(self.entry_input.get())
        self.output_num = int(self.entry_output.get())
        self.hidden_layers = list(map(int, self.entry_hidden.get().split(',')))
        self.learning_rate = float(self.entry_lr.get())
        self.momentum = float(self.entry_momentum.get())
        self.epochs = int(self.entry_epochs.get())
        self.error = float(self.entry_error.get())
        self.use_bias = bool(self.entry_use_bias.get())
        self.log_rate = int(self.entry_log_rate.get())
        self.log_path = str(self.entry_log_path.get())
        self.save_path = str(self.entry_save_path.get())

    def experiment(self):
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
                      dataset=self.patterns,
                      save_path=self.save_path)

            net.train()
            for x, _ in self.patterns:
                net.forward(x)
                h = net.activations[1]  # warstwa ukryta
                print(f"{x} -> hidden = {np.round(h, 4)}")
            if self.log_path is not None:
                data = np.loadtxt(self.log_path, delimiter=",", skiprows=1)
                epochs = data[:, 0].astype(int)
                errors = data[:, 1]
                draw_learning_plot(errors, epochs)
            else:
                print("Brak ścieżki do log file = brak wykresu")

        except ValueError as ve:
            print(f"Invalid entry: {ve}")

    def benchmark(self):
        combos = [
            (0.9, 0.0), (0.6, 0.0), (0.2, 0.0),
            (0.9, 0.6), (0.2, 0.9)
        ]

        output = []

        print(f"\n=== Benchmark (bias={'ON' if self.use_bias else 'OFF'}) ===")
        for lr, mu in combos:
            net = MLP(input_number=self.input_num,
                      output_number=self.output_num,
                      hidden_layers=self.hidden_layers,
                      use_bias=self.use_bias,
                      learning_rate=self.learning_rate,
                      momentum=self.momentum,
                      max_epochs=self.epochs,
                      target_error=self.error,
                      log_rate=self.log_rate,
                      log_path='benchmark.txt',
                      dataset=self.patterns,
                      save_path=None)

            print(f"\n-- η={lr}, μ={mu} --")
            output.append(f"\n-- η={lr}, μ={mu} --")
            net.train()  # w logach zobaczymy, przy której epoce osiąga target_error
            data = np.loadtxt(self.log_path, delimiter=",", skiprows=1)
            epochs = data[:, 0].astype(int)
            errors = data[:, 1]
            draw_learning_plot(errors, epochs)

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


if __name__ == "__main__":
    root_window = tk.Tk()
    app = Encoder(root_window)
    root_window.mainloop()
