import solver

if __name__ == "__main__":
    shuffled_puzzle = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [0, 10, 11, 12],
        [9, 13, 14, 15]
    ]

    order = ["L", "D", "R", "U"]

    print(solver.bfs(shuffled_puzzle, order))
    print(solver.dfs(shuffled_puzzle, order, 20))
    print(solver.a_star(shuffled_puzzle, "hamm"))