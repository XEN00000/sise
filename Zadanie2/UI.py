import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from encoder import experiment
from encoder import benchmark
from mlp import MLP
import dataLoader as dl
import test


def start_training():
    try:
        # Pobranie danych z GUI
        lr = float(learning_rate_entry.get())
        momentum = float(momentum_entry.get())
        hidden_text = hidden_layers_entry.get()
        hidden_layers = [int(n.strip()) for n in hidden_text.split(',') if n.strip().isdigit()]
        use_bias = bias_var.get() == 1

        # Załaduj dane (irysy)
        if not csv_path:
            messagebox.showerror("Błąd", "Wybierz plik CSV przed treningiem.")
            return
        train_dataset, _ = dl.loader(csv_path)

        # Stwórz sieć
        net = MLP(
            input_number=4,
            output_number=3,
            hidden_layers=hidden_layers,
            use_bias=use_bias,
            learning_rate=lr,
            momentum=momentum,
            max_epochs=500,
            target_error=0.001,
            log_rate=50,
            log_path="train_log.csv",
            dataset=train_dataset,
            save_path="trained_gui_model.net"
        )

        # Trenuj
        net.train()
        messagebox.showinfo("Sukces", "Trening zakończony!")

    except Exception as e:
        messagebox.showerror("Błąd", f"Nie udało się przeprowadzić treningu: {e}")


def start_testing():
    try:
        # Wczytaj dane testowe
        if not csv_path:
            messagebox.showerror("Błąd", "Wybierz plik CSV przed testowaniem.")
            return
        _, test_dataset = dl.loader(csv_path)

        # Ścieżka do wytrenowanej sieci
        if not model_path:
            messagebox.showerror("Błąd", "Wybierz plik z modelem sieci.")
            return

        # Użyj istniejącej funkcji testującej, ale przekieruj dane do stringa
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        test.test_the_network(test_dataset, network_path=model_path)

        sys.stdout = old_stdout
        result = mystdout.getvalue()
        messagebox.showinfo("Wyniki testu", result)

    except Exception as e:
        messagebox.showerror("Błąd", f"Nie udało się przetestować sieci: {e}")


def choose_csv():
    global csv_path
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if path:
        csv_path = path
        csv_label.config(text=f"Wybrano: {path.split('/')[-1]}")


def choose_model():
    global model_path
    path = filedialog.askopenfilename(filetypes=[("Network files", "*.ntwrk")])
    if path:
        model_path = path
        model_label.config(text=f"Wybrano: {path.split('/')[-1]}")


def run_autoencoder():
    use_bias = ae_bias_var.get() == 1
    try:
        result = experiment(use_bias)
        messagebox.showinfo("Wyniki autoenkodera", result)
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd autoenkodera: {e}")


def run_benchmark():
    use_bias = ae_bias_var.get() == 1
    try:
        result = benchmark(use_bias)
        messagebox.showinfo("Wyniki autoenkodera", result)
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd autoenkodera: {e}")


# Główne okno
root = tk.Tk()
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Ramki dla każdej zakładki
train_tab = tk.Frame(notebook)
test_tab = tk.Frame(notebook)
auto_tab = tk.Frame(notebook)

# Dodanie zakładek
notebook.add(train_tab, text="Trening")
notebook.add(test_tab, text="Testowanie")
notebook.add(auto_tab, text="Autoenkoder")

# Setting up
model_path = None
csv_path = None
root.title("MLP - Multilayer Perceptron")
root.geometry("320x520")

# Learning rate
tk.Label(train_tab, text="Learning rate (η):").pack(pady=5)
learning_rate_entry = tk.Entry(train_tab)
learning_rate_entry.pack()
learning_rate_entry.insert(0, "0.1")

# Momentum
tk.Label(train_tab, text="Momentum (μ):").pack(pady=5)
momentum_entry = tk.Entry(train_tab)
momentum_entry.pack()
momentum_entry.insert(0, "0.0")

# Bias
bias_var = tk.IntVar(value=1)
bias_checkbox = tk.Checkbutton(train_tab, text="Użyj biasu", variable=bias_var)
bias_checkbox.pack(pady=5)

# Wybór zbioru danych
tk.Button(train_tab, text="Wybierz plik CSV", command=choose_csv).pack(pady=5)
csv_label = tk.Label(train_tab, text="Brak pliku")
csv_label.pack()

# Wybór modelu
tk.Button(test_tab, text="Wybierz plik modelu (*.net)", command=choose_model).pack(pady=5)
model_label = tk.Label(test_tab, text="Brak pliku")
model_label.pack()

# Warstwy ukryte
tk.Label(train_tab, text="Warstwy ukryte (np. 5,4):").pack(pady=5)
hidden_layers_entry = tk.Entry(train_tab)
hidden_layers_entry.pack()
hidden_layers_entry.insert(0, "5,4")

# Przycisk - "Trenuj sieć"
train_button = tk.Button(train_tab, text="Trenuj sieć", command=start_training)
train_button.pack(pady=20)

# Przycisk - "Przetestuj sieć"
test_button = tk.Button(test_tab, text="Przetestuj sieć", command=start_testing)
test_button.pack(pady=5)

# Zakładka Autoenkoder – GUI
ae_bias_var = tk.IntVar(value=1)
tk.Checkbutton(auto_tab, text="Użyj biasu", variable=ae_bias_var).pack(pady=10)

tk.Button(auto_tab, text="Trenuj autoenkoder", command=run_autoencoder).pack(pady=10)
tk.Button(auto_tab, text="Benchmarkuj kombinacje", command=run_benchmark).pack(pady=10)

# Start GUI
root.mainloop()
