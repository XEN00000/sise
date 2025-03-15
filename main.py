#
#
corePuzzle = [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 0, 11, 12],
              [13, 10, 14, 15]]

maxDepthLevel = 20
maxMoves = 5


def swapField(row, col, puzzle):
    if (row < 0 or row > 3) and (col < 0 or col > 3):
        return

    zeroPosition = findBlank(puzzle)
    swappingPosition = row, col

    x = puzzle[zeroPosition]
    puzzle[zeroPosition] = puzzle[swappingPosition]
    puzzle[swappingPosition] = x
    return puzzle


def findBlank(puzzle):
    for row in puzzle:
        for col in row:
            if puzzle[row][col] == 0:
                return row, col


def makeMove(puzzle, move):
    move = move.lower()
    row, col = findBlank(puzzle)
    match move:
        case 'r':
            swappedPuzzle = swapField(row + 1, col, puzzle)
            if swappedPuzzle == puzzle:
                print("Cant make move")
                return
            else:
                return swappedPuzzle

        case 'l':
            swappedPuzzle = swapField(row - 1, col, puzzle)
            if swappedPuzzle == puzzle:
                print("Cant make move")
                return
            else:
                return swappedPuzzle

        case 'u':
            swappedPuzzle = swapField(row, col - 1, puzzle)
            if swappedPuzzle == puzzle:
                print("Cant make move")
                return
            else:
                return swappedPuzzle

        case 'd':
            swappedPuzzle = swapField(row, col + 1, puzzle)
            if swappedPuzzle == puzzle:
                print("Cant make move")
                return
            else:
                return swappedPuzzle
