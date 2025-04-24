Task Summary (English)
The assignment focuses on solving the "Fifteen Puzzle" using a program that transforms a given initial puzzle state into the goal state by applying various state-space search strategies.


Programming Part:
Objective: Implement a program that solves the Fifteen Puzzle using the following search strategies:

-Breadth-first search (bfs)
-Depth-first search (dfs) with a recursion depth limit (minimum 20)
-A* search (astr) with two heuristics:
-Hamming distance (hamm)
-Manhattan distance (manh)


Inputs/Outputs:
- The program takes the following arguments:
	1. Search strategy acronym
	2. Strategy parameter (neighbor order or heuristic)
	3. Input file (initial puzzle state)
	4. Output file (solution)
	5. Output file (additional statistics)


Input File: Specifies the board size (rows and columns) and puzzle layout (0 = empty space).

- Solution File:
	- Line 1: Number of moves (-1 if no solution)
	- Line 2: Sequence of moves using letters L, R, U, D (Left, Right, Up, Down)

- Stats File:
	- Line 1: Length of solution
	- Line 2: Number of visited states
	- Line 3: Number of processed states
	- Line 4: Maximum recursion depth
	- Line 5: Computation time in milliseconds (3 decimal places)




Research Part:
Scope:
Test 413 puzzle configurations with distances from 1 to 7 from the goal state.

Search Variants to Analyze:
- bfs and dfs: Use 8 different neighbor-order permutations (e.g., RDUL, LUDR, etc.)
- astr: Use both Hamming and Manhattan heuristics

Analysis:
- Compare the efficiency of each search method based on data from the stats files.
- Visualize the results using charts (presentation in other formats will reduce the grade).
- Draw conclusions based on the results.

Notes:
	- The program must be universal and support puzzle boards of any size, including non-square ones.

	- Example usage:
	   main.exe bfs RDUL input.txt solution.txt stats.txt