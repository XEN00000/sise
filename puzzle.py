import copy
from collections import deque

# Stan docelowy układanki
goal_puzzle = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 0]
]

def print_table(table):
    for row in table:
        print(row)

def find_zero(puzzle):
    for i, row in enumerate(puzzle):
        for j, value in enumerate(row):
            if value == 0:
                return i, j

def get_possible_moves(puzzle):
    row, col = find_zero(puzzle)
    moves = {}
    if row > 0:
        moves["U"] = (row - 1, col)
    if row < 3:
        moves["D"] = (row + 1, col)
    if col > 0:
        moves["L"] = (row, col - 1)
    if col < 3:
        moves["R"] = (row, col + 1)
    return list(moves.keys())

def do_the_move(move, puzzle):
    new_puzzle = copy.deepcopy(puzzle)
    row, col = find_zero(new_puzzle)
    if move == "U":
        new_puzzle[row][col], new_puzzle[row - 1][col] = new_puzzle[row - 1][col], new_puzzle[row][col]
    elif move == "D":
        new_puzzle[row][col], new_puzzle[row + 1][col] = new_puzzle[row + 1][col], new_puzzle[row][col]
    elif move == "L":
        new_puzzle[row][col], new_puzzle[row][col - 1] = new_puzzle[row][col - 1], new_puzzle[row][col]
    elif move == "R":
        new_puzzle[row][col], new_puzzle[row][col + 1] = new_puzzle[row][col + 1], new_puzzle[row][col]
    return new_puzzle

def bfs(puzzle):
    if puzzle == goal_puzzle:
        return puzzle
    queue = deque([puzzle])
    while queue:
        current = queue.popleft()
        for move in get_possible_moves(current):
            new_state = do_the_move(move, current)
            if new_state == goal_puzzle:
                return new_state
            queue.append(new_state)
    return None

shuffled_puzzle = [
    [1, 2, 3, 4],
    [5, 0, 6, 8],
    [9, 10, 7, 12],
    [13, 14, 11, 15]
]

print(bfs(shuffled_puzzle))