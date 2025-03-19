import copy
from collections import deque

goal_puzzle = [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 0]]


def print_table(table):
    for row in table:
        print(row)


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

    return list(possible_moves.keys())


def do_the_move(moveSymbol, og_puzzle):
    puzzle = copy.deepcopy(og_puzzle)
    row, col = find_zero(puzzle)

    if moveSymbol == "U":
        puzzle[row][col], puzzle[row - 1][col] = puzzle[row - 1][col], puzzle[row][col]
    elif moveSymbol == "D":
        puzzle[row][col], puzzle[row + 1][col] = puzzle[row + 1][col], puzzle[row][col]
    elif moveSymbol == "L":
        puzzle[row][col], puzzle[row][col - 1] = puzzle[row][col - 1], puzzle[row][col]
    elif moveSymbol == "R":
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

    queue = deque()
    queue.append(puzzle)

    for nest_level in range(0, 5):
        queue_temp = deque()
        for person in queue:
            copy_person = copy.deepcopy(person)
            moves = get_possible_moves(copy_person)
            for move in moves:
                new_puzzle = do_the_move(move, copy_person)
                queue_temp.append(new_puzzle) # Here is the queue problem
                if new_puzzle == goal_puzzle:
                    return new_puzzle
            moves.clear()
        queue = queue_temp


if __name__ == '__main__':
    shuffled_puzzle = [[1, 2, 3, 4],
                       [5, 6, 0, 8],
                       [9, 10, 7, 12],
                       [13, 14, 11, 15]]

    result = bfs(shuffled_puzzle)
    print(result)
