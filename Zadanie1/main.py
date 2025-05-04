import solver
import sys


def write_stats(filePath, data):
    with open(filePath, 'w') as file:
        if data[2] is None:
            file.write("-1\n")
        else:
            file.write(f"{data[2]}\n")

        file.write(f"{data[3]}\n")
        file.write(f"{data[4]}\n")
        file.write(f"{data[5]}\n")
        file.write(f"{data[6]}\n")


def write_solution(filePath, data):
    with open(filePath, 'w') as file:
        if data[0] is None or data[1] is None or data[2] is None:
            file.write("-1\n")
        else:
            file.write(f"{data[2]}\n")
            for letter in data[1]:
                file.write(f"{letter} ")


def main(args):
    if len(args) >= 6:
        raise Exception("Podano za dużo argumentów")

    algorithm = args[0]
    if algorithm == "astr":
        strategy = args[1]
    else:
        strategy = list(args[1].upper())
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
        result = solver.bfs(puzzle, strategy)
    elif algorithm == "dfs":
        result = solver.dfs(puzzle, strategy, 20)
    elif algorithm == "astr":
        result = solver.a_star(puzzle, strategy)
    else:
        print("Nie znaleziono algorytmu")
        return

    write_stats(puzzleStatsFile, result)
    write_solution(puzzleSolFile, result)


if __name__ == "__main__":
    main(sys.argv[1:])
