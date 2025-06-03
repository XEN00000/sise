import pickle

import numpy as np


# Funkcja testująca wytrenowaną sieć na zadanym zbiorze danych
def test_the_network(dataset, network_path='irisNet.ntwrk', log_path=None):
    # Wczytanie wytrenowanej sieci z pliku
    with open(network_path, 'rb') as f:
        loaded_net = pickle.load(f) # załadowanie sieci z pliku

    # Przygotowanie pliku logu (jeśli podano)
    if log_path:
        f_log = open(log_path, 'w')
        f_log.write("input;target;prediction;error;hidden_output;output_weights\n") # nagłówek logu

    # Przetwarzanie zbioru testowego
    y_true = []
    y_pred = []
    for x, y in dataset:
        out = loaded_net.forward(x) # propagacja w przód
        y_pred.append(np.argmax(out))   # przewidziana klasa (indeks największego wyjścia)
        y_true.append(np.argmax(y)) # oczekiwana klasa

        # Obliczenie błędu i zapis wybranych danych do logu
        error_vec = y - out # wektor błędów
        hidden_output = loaded_net.activations[1]  # aktywacje pierwszej warstwy ukrytej
        output_weights = loaded_net.layers[-1].W   # zapisanie oczekiwanej klasy

        if log_path:
            f_log.write(f"{x.tolist()};{y.tolist()};{out.tolist()};"  # zapis wejść, oczekiwanych wyjść, wyjść sieci
                        f"{np.mean(np.abs(error_vec)):.4f};"  # zapis błędu średniego bezwzględnego
                        f"{hidden_output.tolist()};"  # zapis aktywacji warstwy ukrytej
                        f"{output_weights.tolist()}\n")  # zapis wag warstwy wyjściowej

    # Obliczenie macierzy pomyłek
    n_classes = loaded_net.output_number
    conf_mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1

    # Obliczenie metryk klasyfikacji
    accuracy = np.trace(conf_mat) / conf_mat.sum()          # dokładność: suma przekątnej / suma wszystkich elementów
    precision = np.diag(conf_mat) / conf_mat.sum(axis=0)    # precyzja: przekątna / suma kolumn
    recall = np.diag(conf_mat) / conf_mat.sum(axis=1)       # czułość (recall): przekątna / suma wierszy
    f1 = 2 * (precision * recall) / (precision + recall)    # miara F1

    # Wyświetlenie wyników
    print(f'\nDokładność testu: {accuracy:.4f}\n')
    print('Macierz pomyłek:')
    print(conf_mat, '\n')
    for i in range(n_classes):
        print(f'Klasa {i}: Precyzja={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}')

    if log_path:
        f_log.close()
