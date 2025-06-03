import tkinter as tk
from tkinter import filedialog, messagebox
from mlp import MLP
from dataLoader import loader
import threading


class MLP_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP - Multi-Layer Perceptron")
        self.root.geometry("1000x1000")

        self.errors = []

        # Create fields
        self.create_fields()

        # Create plot area
        '''
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        '''

    def create_fields(self):
        fields = [
            ("Input neurons", "entry_input"),
            ("Output neurons", "entry_output"),
            ("Hidden layers (comma-separated)", "entry_hidden"),
            ("Learning rate", "entry_lr"),
            ("Momentum", "entry_momentum"),
            ("Epochs", "entry_epochs"),
            ("Target error", "entry_error"),
            ("Log every N epochs", "entry_log_rate"),
            ("Training data file", "data_path"),
            ("Log file path", "log_path"),
            ("Network save path", "save_path"),
        ]

        for label, attr in fields:
            tk.Label(self.root, text=label).pack()
            setattr(self, attr, tk.Entry(self.root))
            getattr(self, attr).pack()

        self.var_bias = tk.BooleanVar()
        tk.Checkbutton(self.root, text="Use bias", variable=self.var_bias).pack()

        tk.Button(self.root, text="Select data file", command=self.select_data_file).pack()

        tk.Button(self.root, text="Train network", command=self.start_training_thread).pack()

    def select_data_file(self):
        filename = filedialog.askopenfilename()
        self.data_path.delete(0, tk.END)
        self.data_path.insert(0, filename)

    def start_training_thread(self):
        threading.Thread(target=self.train_network).start()

    def train_network(self):
        try:
            input_num = int(self.entry_input.get())
            output_num = int(self.entry_output.get())
            hidden_layers = list(map(int, self.entry_hidden.get().split(',')))
            learning_rate = float(self.entry_lr.get())
            momentum = float(self.entry_momentum.get())
            epochs = int(self.entry_epochs.get())
            error = float(self.entry_error.get())
            use_bias = self.var_bias.get()
            log_rate = int(self.entry_log_rate.get())

            train_data, _ = loader(self.data_path.get())

            net = MLP(input_num, output_num, hidden_layers, use_bias,
                      learning_rate, momentum, epochs, error,
                      log_rate, self.log_path.get(), train_data, self.save_path.get())

            self.errors.clear()

            for epoch in range(epochs):
                total_error = 0
                for x, target in train_data:
                    output = net.forward(x)
                    delta = output - target
                    for i in reversed(range(len(net.layers))):
                        layer = net.layers[i]
                        W_next = None if i == len(net.layers) - 1 else net.layers[i + 1].W
                        delta = layer.backward(delta_next=delta, W_next=W_next,
                                               learning_rate=learning_rate, momentum=momentum)
                    total_error += ((target - output) ** 2).mean()
                avg_error = total_error / len(train_data)

                if epoch % log_rate == 0:
                    self.errors.append(avg_error)
                    self.update_plot()

                if avg_error <= error:
                    messagebox.showinfo("Training complete", f"Stopped at epoch {epoch}")
                    break

        except Exception as e:
            messagebox.showerror("Error", str(e))

'''
    def update_plot(self):
        self.ax.clear()
        self.ax.plot(range(len(self.errors)), self.errors, marker='o')
        self.ax.set_xlabel('Logged Epoch')
        self.ax.set_ylabel('Average Error')
        self.ax.set_title('Training Error')
        self.canvas.draw()
'''