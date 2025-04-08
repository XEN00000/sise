import os
import main

def dataGenerator():
    folder_with_puzzles = "./generatedPuzzles"
    folder_with_solutions = "./generatedSolutions"
    folder_with_stats = "./generatedStats"

    algorithms = ["bfs", "dfs", "astr"]

    permutations = [
        'rdul', 'rdlu',
        'drul', 'drlu',
        'ludr', 'lurd',
        'uldr', 'ulrd'
    ]

    metrics = ['hamm', 'manh']

    for algorithm in algorithms:
        if algorithm == "bfs" or algorithm == "dfs":
            heuristic = permutations
        elif algorithm == "astr":
            heuristic = metrics
        else:
            break

        for permutation in heuristic:
            print("Algorithm:", algorithm, "Permutation:", permutation)
            for filename in os.listdir(folder_with_puzzles):
                file_path = os.path.join(folder_with_puzzles, filename)
                if os.path.isfile(file_path):
                    file_number = filename[4:12]
                    file_sol_name = f"{folder_with_solutions}/{filename[:12]}_{algorithm}_{permutation}_sol.txt"
                    file_stat_name = f"{folder_with_stats}/{filename[:12]}_{algorithm}_{permutation}_stats.txt"

                    arg_package = [algorithm, permutation, file_path[2:], file_sol_name, file_stat_name]
                    main.main(arg_package)

if __name__ == "__main__":
    dataGenerator()