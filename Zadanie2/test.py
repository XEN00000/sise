import pickle

import numpy as np


def test_the_network(dataset, network_path='irisNet.ntwrk'):
    # 3. Wczytanie wytrenowanej sieci
    with open(network_path, 'rb') as f:
        loaded_net = pickle.load(f)

    # 4. Predykcje na zbiorze testowym
    y_true = []
    y_pred = []
    for x, y in dataset:
        out = loaded_net.forward(x)
        y_pred.append(np.argmax(out))
        y_true.append(np.argmax(y))

    # 5. Obliczenie macierzy pomyłek
    n_classes = loaded_net.output_number
    conf_mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1

    # 6. Metryki
    accuracy = np.trace(conf_mat) / conf_mat.sum()
    precision = np.diag(conf_mat) / conf_mat.sum(axis=0)
    recall = np.diag(conf_mat) / conf_mat.sum(axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)

    # 7. Wyświetlenie wyników
    print(f'\nDokładność testu: {accuracy:.4f}\n')
    print('Macierz pomyłek:')
    print(conf_mat, '\n')
    for i in range(n_classes):
        print(f'Klasa {i}: Precyzja={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}')
