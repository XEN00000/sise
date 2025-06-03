from matplotlib import pyplot as plt


def draw_learning_plot(errors, epochs):
    # Rysujemy wykres
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, errors, label="MSE każdej epoki")
    plt.xlabel("Epoka")
    plt.ylabel("Średni błąd (MSE)")
    plt.title("Krzywa błędu w trakcie uczenia")
    plt.grid(True)
    plt.legend()
    plt.show()