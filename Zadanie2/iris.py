import tkinter as tk
import numpy as np
from mlp import MLP
import dataLoader as dl
import test
from plotter import draw_learning_plot


class IrisGUI:
    def __init__(self, root):
        self.entry_use_bias = tk.BooleanVar()
        self.root = root
        self.root.title("MLP - Iris set")
        self.root.geometry("600x600")

        # 1. Wzorce sieci irysów
        self.train_dataset, self.test_dataset = dl.loader()

        self.errors = []

        fields = [
            ("Input neurons", "entry_input", 4),
            ("Output neurons", "entry_output", 3),
            ("Hidden layers (comma-separated)", "entry_hidden", "5,4"),
            ("Learning rate", "entry_lr", 0.1),
            ("Momentum", "entry_momentum", 0.9),
            ("Epochs", "entry_epochs", 500),
            ("Target error", "entry_error", 0.001),
            ("Log every N epochs", "entry_log_rate", 50),
            ("Log file path", "entry_log_path",
             "iris_log.txt"),
            ("Network save path", "entry_save_path",
             "irisNet.ntwrk"),
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
        setup_button = tk.Button(self.root, text="Setup", command=self.net_setup)
        setup_button.pack()
        train_button = tk.Button(self.root, text="Train", command=self.train)
        train_button.pack()
        test_button = tk.Button(self.root, text="Test", command=self.test)
        test_button.pack()

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
                      dataset=self.train_dataset,
                      save_path=self.save_path)
            net.train()
            if self.log_path is not None:
                data = np.loadtxt(self.log_path, delimiter=",", skiprows=1)
                epochs = data[:, 0].astype(int)
                errors = data[:, 1]
                draw_learning_plot(errors, epochs)
            else:
                print("Brak ścieżki do log file = brak wykresu")

        except ValueError as ve:
            print(f"Invalid entry: {ve}")

    def test(self):
        test.test_the_network(self.test_dataset, self.save_path, log_path='iris_test_log.csv')


if __name__ == "__main__":
    root_window = tk.Tk()
    app = IrisGUI(root_window)
    root_window.mainloop()
