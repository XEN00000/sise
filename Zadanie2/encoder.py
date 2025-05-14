import numpy as np
import pickle
from mlp import MLP

# 1. Wzorce autoenkodera
patterns = [
    (np.array([1,0,0,0]), np.array([1,0,0,0])),
    (np.array([0,1,0,0]), np.array([0,1,0,0])),
    (np.array([0,0,1,0]), np.array([0,0,1,0])),
    (np.array([0,0,0,1]), np.array([0,0,0,1])),
]

def experiment(use_bias):
    print(f"\n=== Autoencoder (bias={'ON' if use_bias else 'OFF'}) ===")
    net = MLP(input_number=4,
              output_number=4,
              hidden_layers=[2],
              use_bias=use_bias,
              learning_rate=0.6,
              momentum=0.0,
              max_epochs=1000,
              target_error=1e-4,
              log_rate=100,
              log_path=None,
              dataset=patterns,
              save_path=None)
    net.train()
    # Wypisz ukrytą reprezentację
    for x, _ in patterns:
        net.forward(x)
        h = net.activations[1]   # warstwa ukryta
        print(f"{x} -> hidden = {np.round(h, 4)}")


def benchmark(use_bias):
    combos = [
        (0.9, 0.0), (0.6, 0.0), (0.2, 0.0),
        (0.9, 0.6), (0.2, 0.9),
    ]
    print(f"\n=== Benchmark (bias={'ON' if use_bias else 'OFF'}) ===")
    for lr, mu in combos:
        net = MLP(4, 4, [2], use_bias,
                  learning_rate=lr,
                  momentum=mu,
                  max_epochs=2000,
                  target_error=1e-4,
                  log_rate=500,
                  log_path=None,
                  dataset=patterns,
                  save_path=None)
        print(f"\n-- η={lr}, μ={mu} --")
        net.train()   # w logach zobaczysz, przy której epoce osiąga target_error

if __name__ == "__main__":
    # 1) porównujemy bias on/off
    experiment(use_bias=True)
    experiment(use_bias=False)

    # 2) wybierz wersję, która się zbiega, i ją benchmarkuj
    #    zakładam, że bias=True dał zbieżność:
    benchmark(use_bias=True)
