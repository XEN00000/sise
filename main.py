goal_puzzle = [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 0]]


def get_possible_moves(puzzle):
    row, col = find_zero(puzzle)

    possible_moves = {}

    if row > 0:
        possible_moves["U"] = (row - 1, col)  # Can move up?
    if row < 3:
        possible_moves["D"] = (row + 1, col)  # Can move down?
    if col > 0:
        possible_moves["L"] = (row, col - 1)  # Can move left?
    if col < 3:
        possible_moves["R"] = (row, col + 1)  # Can move right?

    print(list(possible_moves.keys()))
    return possible_moves


def do_the_move(moveSymbol, puzzle):
    row, col = find_zero(puzzle)

    if moveSymbol == "U":
        puzzle[row][col], puzzle[row - 1][col] = puzzle[row - 1][col], puzzle[row][col]
    if moveSymbol == "D":
        puzzle[row][col], puzzle[row + 1][col] = puzzle[row + 1][col], puzzle[row][col]
    if moveSymbol == "L":
        puzzle[row][col], puzzle[row][col - 1] = puzzle[row][col - 1], puzzle[row][col]
    if moveSymbol == "R":
        puzzle[row][col], puzzle[row][col + 1] = puzzle[row][col + 1], puzzle[row][col]

    return puzzle


def find_zero(puzzle):
    x, y = 0, 0
    while x <= 3:
        while y <= 3:
            if puzzle[x][y] == 0:
                return x, y
            y += 1
        y = 0
        x += 1


def bfs(puzzle):
    if puzzle == goal_puzzle:
        return puzzle

    nest_level = 0
    tab_of_levels = list()
    tab_of_levels.append(puzzle)
    tab_of_levels.append(puzzle)
    print(tab_of_levels)

    while puzzle != goal_puzzle and nest_level < 5:
        nest_level += 1
        actual_level = list()


if __name__ == '__main__':
    shuffled_puzzle = [[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 0, 15]]

    bfs(shuffled_puzzle)
