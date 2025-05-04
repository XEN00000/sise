import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import LogFormatter


def extract_info_from_name(filename):
    parts = filename.split('_')
    size = parts[0]
    sol_len = int(parts[1])
    control_number = parts[2]
    algorithm = parts[3]
    variant = parts[4]
    return size, sol_len, control_number, algorithm, variant


def load_stats_from_folder(folder):
    data = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('stats.txt'):
                filepath = os.path.join(root, file)
                size, depth, control_number, algo, variant = extract_info_from_name(file)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 5:
                        continue
                    solution_len = int(lines[0])
                    visited = int(lines[1])
                    processed = int(lines[2])
                    max_depth = int(lines[3])
                    time_ms = float(lines[4])
                    data.append({
                        'size': size,
                        'depth': depth,
                        'control_number': control_number,
                        'algorithm': algo,
                        'variant': variant,
                        'solution_len': solution_len,
                        'visited': visited,
                        'processed': processed,
                        'max_depth': max_depth,
                        'time_ms': time_ms
                    })
    return pd.DataFrame(data)


def plot_metric_over_depth(df, metric, ylabel='Kryterium', main_title=None):
    depths = sorted(df['depth'].unique())
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    if main_title:
        fig.suptitle(main_title, fontsize=16, y=0.98)

    bar_width = 0.25

    # --- Wykres 1: Ogółem BFS/DFS/A* ---
    ax = axs[0, 0]
    algos = ['bfs', 'dfs', 'astr']
    shifts = [-bar_width, 0, bar_width]
    labels = ['BFS', 'DFS', 'A*']
    all_values = []
    for algo, shift, label in zip(algos, shifts, labels):
        values = [np.mean(df[(df['algorithm'] == algo) & (df['depth'] == d)][metric]) for d in depths]
        all_values.extend(values)
        ax.bar(np.array(depths) + shift, values, width=bar_width, label=label)
    if should_use_log_scale(all_values):
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    ax.set_xticks(depths)
    ax.set_title("Ogółem")
    ax.set_xlabel("Głębokość")
    ax.set_ylabel(ylabel)
    ax.legend()

    # --- Wykres 2: A* - heurystyki ---
    ax = axs[0, 1]
    astr = df[df['algorithm'] == 'astr']
    variants = ['hamm', 'manh']
    shifts = [-bar_width / 2, bar_width / 2]
    labels = ['Hamming', 'Manhattan']
    all_values = []
    for variant, shift, label in zip(variants, shifts, labels):
        values = [np.mean(astr[(astr['variant'] == variant) & (astr['depth'] == d)][metric]) for d in depths]
        all_values.extend(values)
        ax.bar(np.array(depths) + shift, values, width=bar_width, label=label)
    if should_use_log_scale(all_values):
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    ax.set_xticks(depths)
    ax.set_title("A*")
    ax.set_xlabel("Głębokość")
    ax.set_ylabel(ylabel)
    ax.legend()

    # --- Wykres 3: BFS - porządki ---
    ax = axs[1, 0]
    bfs = df[df['algorithm'] == 'bfs']
    variants = sorted(bfs['variant'].unique())
    width = 0.8 / len(variants)
    shifts = np.linspace(-0.4, 0.4, len(variants))
    all_values = []
    for variant, shift in zip(variants, shifts):
        values = [np.mean(bfs[(bfs['variant'] == variant) & (bfs['depth'] == d)][metric]) for d in depths]
        all_values.extend(values)
        ax.bar(np.array(depths) + shift, values, width=width, label=variant.upper())
    if should_use_log_scale(all_values):
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    ax.set_xticks(depths)
    ax.set_title("BFS")
    ax.set_xlabel("Głębokość")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

    # --- Wykres 4: DFS - porządki ---
    ax = axs[1, 1]
    dfs = df[df['algorithm'] == 'dfs']
    variants = sorted(dfs['variant'].unique())
    width = 0.8 / len(variants)
    shifts = np.linspace(-0.4, 0.4, len(variants))
    all_values = []
    for variant, shift in zip(variants, shifts):
        values = [np.mean(dfs[(dfs['variant'] == variant) & (dfs['depth'] == d)][metric]) for d in depths]
        all_values.extend(values)
        ax.bar(np.array(depths) + shift, values, width=width, label=variant.upper())
    if should_use_log_scale(all_values):
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    ax.set_xticks(depths)
    ax.set_title("DFS")
    ax.set_xlabel("Głębokość")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def should_use_log_scale(values, threshold=100):
    values = [v for v in values if v > 0]
    if not values or min(values) == 0:
        return False
    return max(values) / min(values) > threshold


df = load_stats_from_folder('./generatedStats')

print(df)

# TODO: poprawić etykiety na wykresach. Aktualnie pokazywanie czasu jest nieczytelne i wymaga poprawy

plot_metric_over_depth(df, 'time_ms', ylabel='Czas w sekundach', main_title='Czas działania algorytmu')
'''
plot_metric_over_depth(df, 'solution_len', main_title='Długość rozwiązania')
plot_metric_over_depth(df, 'visited', main_title='Liczba stanów odwiedzonych')
plot_metric_over_depth(df, 'processed', main_title='Liczba stanów przetworzonych')
plot_metric_over_depth(df, 'max_depth', main_title='Maksymalna głębokość rekursji')
'''
