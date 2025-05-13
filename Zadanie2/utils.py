import csv
import pickle


def load_dataset(csv_path, delimiter=',', skip_header=False):
    raw_inputs = []
    raw_labels = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            vals = [float(x) for x in row]
            raw_inputs.append(vals[:-1])  # pierwsze 4 kolumny
            raw_labels.append(int(vals[-1]))  # ostatnia kolumna
    return raw_inputs, raw_labels


def prepare_dataset(raw_inputs, raw_labels, n_classes):
    # 1) oblicz min/max per kolumna
    cols = list(zip(*raw_inputs))
    mins = [min(col) for col in cols]
    maxs = [max(col) for col in cols]

    def normalize(inp):
        return [
            (x - mins[i]) / (maxs[i] - mins[i]) if maxs[i] != mins[i] else 0.0
            for i, x in enumerate(inp)
        ]

    data = []
    for inp, lab in zip(raw_inputs, raw_labels):
        ni = normalize(inp)
        # one-hot
        tgt = [0.0] * n_classes
        tgt[lab] = 1.0
        data.append((ni, tgt))
    return data


def save_network(network, filepath):
    """
    Zapisuje obiekt sieci (instancję klasy Network) do pliku binarnego.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(network, f)
    print(f"Sieć zapisana do: {filepath}")


def load_network(filepath):
    """
    Wczytuje obiekt sieci z pliku zapisanego przez save_network.
    """
    with open(filepath, 'rb') as f:
        net = pickle.load(f)
    print(f"Sieć wczytana z: {filepath}")
    return net


def init_error_log(log_path, header="epoch,error"):
    """
    Tworzy (lub nadpisuje) plik z nagłówkiem do logowania błędu globalnego.
    """
    with open(log_path, 'w') as f:
        f.write(header + "\n")
    print(f"Plik logu błędu utworzony: {log_path}")


def append_error_log(log_path, epoch, error):
    """
    Dopisuje do pliku logu kolejną linię w formacie: epoch,error
    """
    with open(log_path, 'a') as f:
        f.write(f"{epoch},{error}\n")
