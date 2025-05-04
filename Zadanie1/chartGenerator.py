import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

STATS_FOLDER = "./generatedStats"

ALGORITHMS = ["bfs", "dfs", "astr"]
#ALGORITHMS = ["dfs"]
PERMUTATIONS = ["rdul", "rdlu", "drul", "drlu", "ludr", "lurd", "uldr", "ulrd"]
HEURISTICS = ["hamm", "manh"]

METRICS = {
    0: "Długość rozwiązania",
    1: "Stany odwiedzone",
    2: "Stany przetworzone",
    3: "Maks. głębokość rekursji",
    4: "Czas [s]"
}

def generate_charts():
    metrics_data = {i: defaultdict(lambda: defaultdict(list)) for i in range(5)}

    for filename in os.listdir(STATS_FOLDER):
        if not filename.endswith("_stats.txt"):
            continue

        parts = filename[:-10].split("_")
        if "dfs" in parts:
            algo_index = parts.index("dfs")
        elif "bfs" in parts:
            algo_index = parts.index("bfs")
        elif "astr" in parts:
            algo_index = parts.index("astr")
        else:
            continue

        algo = parts[algo_index]
        variant = parts[algo_index + 1] if algo_index + 1 < len(parts) else ""

        filepath = os.path.join(STATS_FOLDER, filename)

        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
                if len(lines) < 5:
                    continue
                depth = int(lines[0])
                if depth <= 0:
                    continue
                for i in range(5):
                    val = float(lines[i].strip())
                    key = algo
                    if algo == "dfs" or algo == "bfs":
                        key = f"{algo.upper()}-{variant.upper()}"
                    elif algo == "astr":
                        key = f"A*-{variant.capitalize()}"
                    metrics_data[i][key][depth].append(val)
        except:
            continue

    os.makedirs("plots", exist_ok=True)

    for metric_id, metric_name in METRICS.items():
        data = metrics_data[metric_id]
        general = defaultdict(list)
        by_bfs = defaultdict(list)
        by_dfs = defaultdict(list)
        by_astar = defaultdict(list)
        depths = range(1, 8)

        for d in depths:
            for group, prefix in [(general, "BFS-"), (general, "DFS-"), (general, "A*-")]:
                values = []
                for key in data:
                    if key.startswith(prefix):
                        values.extend(data[key][d])
                group[prefix[:-1]].append(sum(values)/len(values) if values else 0)

            for perm in PERMUTATIONS:
                bfs_key = f"BFS-{perm.upper()}"
                dfs_key = f"DFS-{perm.upper()}"
                bfs_vals = data[bfs_key][d]
                dfs_vals = data[dfs_key][d]
                by_bfs[perm.upper()].append(sum(bfs_vals)/len(bfs_vals) if bfs_vals else 0)
                by_dfs[perm.upper()].append(sum(dfs_vals)/len(dfs_vals) if dfs_vals else 0)

            for h in HEURISTICS:
                key = f"A*-{h.capitalize()}"
                vals = data[key][d]
                by_astar[h.capitalize()].append(sum(vals)/len(vals) if vals else 0)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(metric_name)

        x = np.arange(len(depths))
        width = 0.25

        # Ogólne BFS/DFS/A*
        axs[0, 0].bar(x - width, general["BFS"], width=width, label="BFS")
        axs[0, 0].bar(x, general["DFS"], width=width, label="DFS")
        axs[0, 0].bar(x + width, general["A*"], width=width, label="A*")
        axs[0, 0].set_title("Ogółem")
        axs[0, 0].set_ylabel("Kryterium")
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(depths)
        axs[0, 0].legend()

        # A*
        axs[0, 1].bar(x - width / 2, by_astar["Hamm"], width=width, label="Hamming")
        axs[0, 1].bar(x + width / 2, by_astar["Manh"], width=width, label="Manhattan")
        axs[0, 1].set_title("A*")
        axs[0, 1].set_xticks(x)
        axs[0, 1].set_xticklabels(depths)
        axs[0, 1].legend()

        # BFS permutacje
        width_bfs = 0.08
        for i, (perm, vals) in enumerate(by_bfs.items()):
            axs[1, 0].bar(x + i * width_bfs, vals, width=width_bfs, label=perm)
        axs[1, 0].set_title("BFS")
        axs[1, 0].set_xlabel("Głębokość")
        axs[1, 0].set_xticks(x + width_bfs * (len(by_bfs) / 2))
        axs[1, 0].set_xticklabels(depths)
        axs[1, 0].legend(fontsize=6)

        # DFS permutacje
        width_dfs = 0.08
        for i, (perm, vals) in enumerate(by_dfs.items()):
            axs[1, 1].bar(x + i * width_dfs, vals, width=width_dfs, label=perm)
        axs[1, 1].set_title("DFS")
        axs[1, 1].set_xlabel("Głębokość")
        axs[1, 1].set_xticks(x + width_dfs * (len(by_dfs) / 2))
        axs[1, 1].set_xticklabels(depths)
        axs[1, 1].legend(fontsize=6)

        plt.tight_layout()
        plt.savefig(f"plots/kryterium_{metric_id + 1}.png")
        plt.close()

if __name__ == "__main__":
    generate_charts()
