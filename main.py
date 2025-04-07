import solver
import sys


def main(args):
    if len(args) > 5:
        raise Exception("Podano za dużo argumentów")

    algorithm = args[0]
    strategy = args[1]
    puzzleFile = args[2]
    puzzleSolFile = args[3]
    puzzleStatsFile = args[4]

    puzzle = []

    try:
        with open(puzzleFile, 'r') as file:
            next(file)
            for line in file:
                numbers = list(map(int, line.strip().split()))
                puzzle.append(numbers)
    except FileNotFoundError:
        print("Nie znaleziono pliku o nazwie:", puzzleFile)
    except Exception as e:
        print("Wystąpił błąd:", e)

    if algorithm == "bfs":
        result = solver.dfs(puzzle, strategy, 5)
    elif algorithm == "dfs":
        result = solver.bfs(puzzle, strategy)
    elif algorithm == "astr":
        result = solver.a_star(puzzle, strategy)
    else:
        print("Nie znaleziono algorytmu")
        return

    with open(puzzleSolFile, 'w') as file:
        if result[1] is None or result[2] is None:
            file.write("-1\n")
        else:
            file.write(f"{result[2]}\n")
            for letter in result[1]:
                file.write(f"{letter} ")

    with open(puzzleStatsFile, 'w') as file:
        if result[2] is None:
            file.write("-1\n")
        else:
            file.write(f"{result[2]}\n")
        file.write(f"{result[3]}\n")
        file.write(f"{result[4]}\n")
        file.write(f"{result[5]}\n")
        file.write(f"{result[6]}\n")


if __name__ == "__main__":
    main(sys.argv)
