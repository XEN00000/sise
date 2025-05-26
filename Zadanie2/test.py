import pickle

import numpy as np


def test_the_network(dataset, network_path='irisNet.ntwrk', log_path=None):
    # 3. Wczytanie wytrenowanej sieci
    with open(network_path, 'rb') as f:
        loaded_net = pickle.load(f)

    if log_path:
        f_log = open(log_path, 'w')
        f_log.write("input,target,prediction,error,hidden_output,output_weights\n")

    # 4. Predykcje na zbiorze testowym
    y_true = []
    y_pred = []
    for x, y in dataset:
        out = loaded_net.forward(x)
        y_pred.append(np.argmax(out))
        y_true.append(np.argmax(y))

        error_vec = y - out
        hidden_output = loaded_net.activations[1]  # pierwsza warstwa ukryta
        output_weights = loaded_net.layers[-1].W

        if log_path:
            f_log.write(f"{x.tolist()},{y.tolist()},{out.tolist()},"
                        f"{np.mean(np.abs(error_vec)):.4f},"
                        f"{hidden_output.tolist()},"
                        f"{output_weights.tolist()}\n")

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

    if log_path:
        f_log.close()
