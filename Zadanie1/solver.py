# Program ma generować informacje obejmujące:
# -długość znalezionego rozwiązania;
# -liczbę stanów odwiedzonych;
# -liczbę stanów przetworzonych;
# -maksymalną osiągniętą głębokość rekursji;
# -czas trwania procesu obliczeniowego.

import copy
import heapq
import time
from collections import deque

# Stan docelowy układanki
goal_puzzle = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 0]
]

goal_positions = {}
for i, row in enumerate(goal_puzzle):
    for j, tile in enumerate(row):
        goal_positions[tile] = (i, j)


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
        elapsed = round(end_time - start_time, 6)
        return puzzle, [], 0, 1, 0, 1, elapsed

    queue = deque([(puzzle, [])])
    visited_states = 0
    processed_states = 0
    max_depth_reached = 1
    visited = set()

    while queue:
        current, path = queue.popleft()
        processed_states += 1

        if current == goal_puzzle:
            end_time = time.time()
            elapsed = round(end_time - start_time, 3)
            return current, path, len(path), visited_states, processed_states, max_depth_reached, elapsed

        possible_moves = get_possible_moves(current)
        ordered_moves = [move for move in move_order if move in possible_moves]

        for move in ordered_moves:
            new_state = do_the_move(move, current)
            new_path = path + [move]

            new_tuple = tuple(tuple(row) for row in new_state)
            if new_tuple in visited:
                continue

            visited.add(new_tuple)
            queue.append((new_state, new_path))
            visited_states += 1
            max_depth_reached = max(max_depth_reached, len(new_path))

    end_time = time.time()
    elapsed = round(end_time - start_time, 3)
    return None, None, None, visited_states, processed_states, max_depth_reached, elapsed


def dfs(puzzle, move_order, max_depth):
    start_time = time.time()

    if puzzle == goal_puzzle:
        end_time = time.time()
        elapsed = round(end_time - start_time, 3)
        return puzzle, [], 0, 1, 0, 1, elapsed

    stack = deque([(puzzle, [])])
    visited_states = 0
    processed_states = 0
    max_depth_reached = 1
    visited = dict()

    while stack:
        current, path = stack.pop()
        processed_states += 1

        if current == goal_puzzle:
            end_time = time.time()
            elapsed = round(end_time - start_time, 3)
            return current, path, len(path), visited_states, processed_states, max_depth_reached, elapsed

        if len(path) >= max_depth:
            max_depth_reached = max(max_depth_reached, len(path))
            continue

        current_depth = len(path)
        current_tuple = tuple(tuple(row) for row in current)
        if current_tuple in visited and visited[current_tuple] < current_depth:
            continue
        visited[current_tuple] = current_depth

        possible_moves = get_possible_moves(current)
        ordered_moves = [move for move in move_order if move in possible_moves]

        for move in reversed(ordered_moves):
            if path and ((move == "L" and path[-1] == "R") or
                         (move == "R" and path[-1] == "L") or
                         (move == "U" and path[-1] == "D") or
                         (move == "D" and path[-1] == "U")):
                continue

            new_state = do_the_move(move, current)
            visited_states += 1

            new_path = path + [move]
            max_depth_reached = max(max_depth_reached, len(new_path))

            stack.append((new_state, new_path))

    end_time = time.time()
    elapsed = round(end_time - start_time, 3)
    return None, None, None, visited_states, processed_states, max_depth_reached, elapsed


def manhattan_distance(puzzle):
    distance = 0
    for i, row in enumerate(puzzle):
        for j, tile in enumerate(row):
            if tile != 0:
                goal_i, goal_j = goal_positions[tile]
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance


def hamming_distance(puzzle):
    distance = 0
    for i, row in enumerate(puzzle):
        for j, tile in enumerate(row):
            if tile != 0 and tile != goal_puzzle[i][j]:
                distance += 1
    return distance


def a_star(puzzle, heuristic):
    start_time = time.time()

    if puzzle == goal_puzzle:
        end_time = time.time()
        elapsed = round(end_time - start_time, 3)
        return puzzle, [], 0, 1, 0, 1, elapsed

    if heuristic == 'manh':
        h = manhattan_distance(puzzle)
    elif heuristic == 'hamm':
        h = hamming_distance(puzzle)
    else:
        raise ValueError("Unknown heuristic format: {}. Use 'manh' or 'hamm'".format(heuristic))

    start_node = (h, 0, puzzle, [])
    frontier = []
    heapq.heappush(frontier, start_node)

    visited = {tuple(tuple(row) for row in puzzle): 0}

    visited_states = 0
    processed_states = 0
    max_depth_reached = 1

    while frontier:
        priority, cost, current, path = heapq.heappop(frontier)
        processed_states += 1

        if current == goal_puzzle:
            end_time = time.time()
            elapsed = round(end_time - start_time, 3)
            return current, path, len(path), visited_states, processed_states, max_depth_reached, elapsed

        for move in get_possible_moves(current):
            new_state = do_the_move(move, current)
            new_path = path + [move]
            visited_states += 1
            max_depth_reached = max(max_depth_reached, len(new_path))
            new_cost = cost + 1

            if heuristic == 'manh':
                h_new = manhattan_distance(new_state)
            if heuristic == 'hamm':
                h_new = hamming_distance(new_state)
            new_priority = new_cost + h_new

            state_tuple = tuple(tuple(row) for row in new_state)  # hashowanie listy(aktualnego stanu) do tupla(krotki)
            if state_tuple not in visited or visited[state_tuple] > new_cost:
                visited[state_tuple] = new_cost  # nadpisujemy koszt a nie tupla
                heapq.heappush(frontier, (new_priority, new_cost, new_state, new_path))

    end_time = time.time()
    elapsed = round(end_time - start_time, 3)
    return None, None, None, visited_states, processed_states, max_depth_reached, elapsed
