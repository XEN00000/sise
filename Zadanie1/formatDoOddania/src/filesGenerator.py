import os
import main
from multiprocessing import Pool

folder_with_puzzles = "./generatedPuzzles"
folder_with_solutions = "./generatedSolutions"
folder_with_stats = "./generatedStats"

# DFS permutacje
dfs_permutations = [
    'rdul', 'rdlu',
    'drul', 'drlu',
    'ludr', 'lurd',
    'uldr', 'ulrd'
]

# Heurystyki do A*
metrics = ['hamm', 'manh']

# Wszystkie puzzle
puzzles = [f for f in os.listdir(folder_with_puzzles) if f.endswith(".txt")]


def run_dfs_for_permutation(permutation):
    counter = 0
    algorithm = "dfs"

    for filename in puzzles:
        file_path = os.path.join(folder_with_puzzles, filename)
        file_base = filename[:12]
        file_sol_name = f"{folder_with_solutions}/{file_base}_{algorithm}_{permutation}_sol.txt"
        file_stat_name = f"{folder_with_stats}/{file_base}_{algorithm}_{permutation}_stats.txt"
        file_count = len([f for f in os.listdir(folder_with_puzzles) if os.path.isfile(os.path.join(folder_with_puzzles, f))])
        counter += 1
        print("Algorithm dfs permutation: ", permutation, "  ", round(counter / file_count * 100, 2), "%")

        try:
            main.main([algorithm, permutation, file_path, file_sol_name, file_stat_name])
        except Exception as e:
            print(f"[DFS-{permutation}] Błąd: {filename} → {e}")
        print("Algorithm dfs permutation: ", permutation, "  ", round(counter / file_count * 100, 2), "%")


def run_other_algorithms():
    # BFS
    for permutation in dfs_permutations:
        print("Algorithm bfs permutation: ", permutation)
        for filename in puzzles:
            file_path = os.path.join(folder_with_puzzles, filename)
            file_base = filename[:12]
            file_sol_name = f"{folder_with_solutions}/{file_base}_bfs_{permutation}_sol.txt"
            file_stat_name = f"{folder_with_stats}/{file_base}_bfs_{permutation}_stats.txt"
            try:
                main.main(["bfs", permutation, file_path, file_sol_name, file_stat_name])
            except Exception as e:
                print(f"[BFS-{permutation}] Błąd: {filename} → {e}")

    # A*
    for metric in metrics:
        print("Algorithm astr metric: ", metric)
        for filename in puzzles:
            file_path = os.path.join(folder_with_puzzles, filename)
            file_base = filename[:12]
            file_sol_name = f"{folder_with_solutions}/{file_base}_astr_{metric}_sol.txt"
            file_stat_name = f"{folder_with_stats}/{file_base}_astr_{metric}_stats.txt"
            try:
                main.main(["astr", metric, file_path, file_sol_name, file_stat_name])
            except Exception as e:
                print(f"[ASTR-{metric}] Błąd: {filename} → {e}")


if __name__ == "__main__":
    os.makedirs(folder_with_solutions, exist_ok=True)
    os.makedirs(folder_with_stats, exist_ok=True)

    # Uruchom BFS i A* normalnie
    run_other_algorithms()

    # Uruchom DFS na 8 wątkach (każdy z osobną permutacją)
    with Pool(processes=8) as pool:
        pool.map(run_dfs_for_permutation, dfs_permutations)
