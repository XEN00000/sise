from UI import *

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


'''
import dataLoader as dl
import mlp
import test

train_dataset, test_dataset = dl.loader()

net = mlp.MLP(input_number=4,
              output_number=3,
              hidden_layers=[5, 4],
              use_bias=True,
              learning_rate=0.1,
              momentum=0.9,
              max_epochs=500,
              target_error=0.001,
              log_rate=50,
              log_path='log.txt',
              dataset=train_dataset,
              save_path='irisNet.ntwrk')

net.train()

test.test_the_network(test_dataset, 'irisNet.ntwrk', log_path='test_log.csv')
'''