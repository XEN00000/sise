import csv
import numpy as np


#def loader(dataset_name='irisset.csv', shuffled=True, training_set_size=0.7):
def loader(dataset_name='irisset.csv'):
    # 1. Wczytujemy z CSV
    data = []
    with open(dataset_name, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # cechy: pierwsze 4 kolumny jako float
            features = list(map(float, row[:4]))
            # etykieta: ostatnia kolumna jako string
            label = row[4]
            data.append((features, label))

    # 2. Zamieniamy etykiety na indeksy i one-hot
    labels = sorted({lab for _, lab in data})
    lab2idx = {lab: i for i, lab in enumerate(labels)}
    n_classes = len(labels)

    X = np.array([f for f, _ in data])
    y_idx = np.array([lab2idx[lab] for _, lab in data])
    Y = np.zeros((len(y_idx), n_classes))
    Y[np.arange(len(y_idx)), y_idx] = 1

    # 3. Standaryzacja cech (scala po wierszach)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # 4. Podział na zbiór treningowy i testowy (70/30)
    perm = np.random.RandomState(42).permutation(len(X))
    cut = int(0.7 * len(X))
    train_idx, test_idx = perm[:cut], perm[cut:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # 5. Listy (wejście, cel) dla MLP
    train_dataset = [(X_train[i], Y_train[i]) for i in range(len(train_idx))]
    test_dataset = [(X_test[i], Y_test[i]) for i in range(len(test_idx))]

    return train_dataset, test_dataset
