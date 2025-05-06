import os
import matplotlib.pyplot as plt

import pandas as pd


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
                        'depth': depth,
                        'algorithm': algo,
                        'variant': variant,
                        'solution_len': solution_len,
                        'visited': visited,
                        'processed': processed,
                        'max_depth': max_depth,
                        'time_ms': time_ms
                    })
    return pd.DataFrame(data)


def add_plot(dataframe, axn, title='Tytuł', xlabel='Głębokość', ylabel=None, time=False):
    min_val = dataframe[dataframe > 0].min().min()
    max_val = dataframe.max().max()
    use_log = bool((min_val > 0) and (max_val / min_val > 100))

    if use_log:
        dataframe.plot(kind='bar', ax=axn, logy=use_log, rot=0)
        axn.set_title(title)
        axn.set_xlabel(xlabel)
    else:
        if not time:
            dataframe.plot(kind='bar', ax=axn, logy=use_log, rot=0)
            axn.set_title(title)
            axn.set_xlabel(xlabel)
            step = int((max_val - min_val) / 5)
            axn.set_yticks(range(int(min_val), int(max_val + 1), step))
        else:
            dataframe.plot(kind='bar', ax=axn, logy=use_log, rot=0)
            axn.set_title(title)
            axn.set_xlabel(xlabel)

    if ylabel is not None:
        axn.set_ylabel(ylabel)


def launch_the_plot(frameOfData, valesType=None, yLabelName='Kryterium'):
    if valesType == 'time_ms':
        df_mean = frameOfData.pivot_table(index='depth', columns='algorithm', values=valesType, aggfunc='mean')
        df_mean = df_mean[['bfs', 'dfs', 'astr']]

        df_astr = (
            frameOfData[frameOfData['algorithm'] == 'astr'].pivot_table(index='depth', columns='variant',
                                                                        values=valesType,
                                                                        aggfunc='mean'))
        df_astr = df_astr[['hamm', 'manh']]

        df_bfs = (
            frameOfData[frameOfData['algorithm'] == 'bfs'].pivot_table(index='depth', columns='variant',
                                                                       values=valesType,
                                                                       aggfunc='mean'))

        df_dfs = (
            frameOfData[frameOfData['algorithm'] == 'dfs'].pivot_table(index='depth', columns='variant',
                                                                       values=valesType,
                                                                       aggfunc='mean'))
    else:
        df_mean = frameOfData.pivot_table(index='depth', columns='algorithm', values=valesType, aggfunc='mean').round()
        df_mean = df_mean[['bfs', 'dfs', 'astr']]

        df_astr = (
            frameOfData[frameOfData['algorithm'] == 'astr'].pivot_table(index='depth', columns='variant',
                                                                        values=valesType,
                                                                        aggfunc='mean').round())
        df_astr = df_astr[['hamm', 'manh']]

        df_bfs = (
            frameOfData[frameOfData['algorithm'] == 'bfs'].pivot_table(index='depth', columns='variant',
                                                                       values=valesType,
                                                                       aggfunc='mean').round())

        df_dfs = (
            frameOfData[frameOfData['algorithm'] == 'dfs'].pivot_table(index='depth', columns='variant',
                                                                       values=valesType,
                                                                       aggfunc='mean').round())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=0.1, hspace=0.3)
    ax1, ax2, ax3, ax4 = axes.flatten()

    if valesType == 'time_ms':
        add_plot(df_mean, ax1, time=True, title='Ogółem', xlabel='Głębokość', ylabel=yLabelName)
        add_plot(df_astr, ax2, time=True, title='A*', xlabel='Głębokość')
        add_plot(df_bfs, ax3, time=True, title='BFS', xlabel='Głębokość', ylabel=yLabelName)
        add_plot(df_dfs, ax4, time=True, title='DFS', xlabel='Głębokość')
    else:
        add_plot(df_mean, ax1, title='Ogółem', xlabel='Głębokość', ylabel=yLabelName)
        add_plot(df_astr, ax2, title='A*', xlabel='Głębokość')
        add_plot(df_bfs, ax3, title='BFS', xlabel='Głębokość', ylabel=yLabelName)
        add_plot(df_dfs, ax4, title='DFS', xlabel='Głębokość')

    plt.show()
    fig.savefig(f'./plots/{valesType}.png', format='png')


if __name__ == "__main__":
    df = load_stats_from_folder('./generatedStats')

    print(df[df['algorithm'] == 'bfs'].pivot_table(index='depth', columns='variant', values='time_ms', aggfunc='mean'))
    print(df[df['algorithm'] == 'dfs'].pivot_table(index='depth', columns='variant', values='time_ms', aggfunc='mean'))
    print(df[df['algorithm'] == 'astr'].pivot_table(index='depth', columns='variant', values='time_ms', aggfunc='mean'))

    launch_the_plot(df, 'solution_len', 'Długość rozwiązania')
    launch_the_plot(df, 'visited', 'Odwiedzone stany')
    launch_the_plot(df, 'processed', 'Przetworzone stany')
    launch_the_plot(df, 'max_depth', 'Maks. głębokość rekursji')
    launch_the_plot(df, 'time_ms', 'Czas działania [s]')
