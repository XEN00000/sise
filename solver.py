# Program ma generować informacje obejmujące:
# -długość znalezionego rozwiązania;
# -liczbę stanów odwiedzonych;
# -liczbę stanów przetworzonych;
# -maksymalną osiągniętą głębokość rekursji;
# -czas trwania procesu obliczeniowego.

import copy
import time
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


def bfs(puzzle, move_order):
    start_time = time.time()

    if puzzle == goal_puzzle:
        end_time = time.time()
        #elapsed = round(end_time - start_time, 6)
        elapsed = end_time - start_time
        return puzzle, [], 0, 1, 0, 1, elapsed

    queue = deque([(puzzle, [])])
    visited_states = 0
    processed_states = 0
    max_depth_reached = 1

    while queue:
        current, path = queue.popleft()
        visited_states += 1

        possible_moves = get_possible_moves(current)
        ordered_moves = [move for move in move_order if move in possible_moves]

        for move in ordered_moves:
            new_state = do_the_move(move, current)
            processed_states += 1
            new_path = path + [move]
            max_depth_reached = max(max_depth_reached, len(new_path))

            if new_state == goal_puzzle:
                end_time = time.time()
                #elapsed = round(end_time - start_time, 6)
                elapsed = end_time - start_time
                return new_state, new_path, len(new_path), visited_states, processed_states, max_depth_reached, elapsed

            queue.append((new_state, new_path))

    end_time = time.time()
    #elapsed = round(end_time - start_time, 6)
    elapsed = end_time - start_time
    return None, None, None, visited_states, processed_states, max_depth_reached, elapsed


def dfs(puzzle, move_order, max_depth):
    start_time = time.time()

    if puzzle == goal_puzzle:
        end_time = time.time()
        #elapsed = round(end_time - start_time, 6)
        elapsed = end_time - start_time
        return puzzle, [], 0, 1, 0, 1, elapsed

    stack = deque([(puzzle, [])])
    visited_states = 0
    processed_states = 0
    max_depth_reached = 1


    while stack:
        current, path = stack.pop()
        visited_states += 1

        if len(path) >= max_depth:
            continue

        possible_moves = get_possible_moves(current)
        ordered_moves = [move for move in move_order if move in possible_moves]

        for move in reversed(ordered_moves):
            new_state = do_the_move(move, current)
            processed_states += 1

            new_path = path + [move]
            max_depth_reached = max(max_depth_reached, len(new_path))

            if new_state == goal_puzzle:
                end_time = time.time()
                #elapsed = round(end_time - start_time, 6)
                elapsed = end_time - start_time
                return new_state, new_path, len(new_path), visited_states, processed_states, max_depth_reached, elapsed

            stack.append((new_state, new_path))

    end_time = time.time()
    #elapsed = round(end_time - start_time, 6)
    elapsed = end_time - start_time
    return None, None, None, visited_states, processed_states, max_depth_reached, elapsed


def a_star():
    pass
