import tkinter as tk
import random
import copy
import puzzle  # Importujemy logikę układanki


class PuzzleGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Układanka Piętnastka")
        self.size = 4
        # Startujemy z ułożoną układanką
        self.board = copy.deepcopy(puzzle.goal_puzzle)
        self.buttons = []
        self.move_count = 0
        self.create_widgets()
        self.update_buttons()
        self.update_counters()

    def create_widgets(self):
        # Ramka na planszę
        self.frame = tk.Frame(self.master)
        self.frame.grid(row=0, column=0, padx=10, pady=10)

        # Tworzymy siatkę przycisków reprezentujących kafelki
        for i in range(self.size):
            row_buttons = []
            for j in range(self.size):
                btn = tk.Button(self.frame, width=4, height=2, font=("Helvetica", 20),
                                command=lambda r=i, c=j: self.on_tile_click(r, c))
                btn.grid(row=i, column=j, padx=2, pady=2)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

        # Przycisk tasowania
        self.shuffle_button = tk.Button(self.master, text="Tasuj", command=self.shuffle)
        self.shuffle_button.grid(row=1, column=0, pady=10)

        # Przycisk rozwiązania (BFS, działa dla maks. 5 ruchów)
        self.solve_button = tk.Button(self.master, text="Rozwiąż (BFS, max 5 ruchów)", command=self.solve)
        self.solve_button.grid(row=2, column=0, pady=10)

        # Etykiety liczników
        #self.move_label = tk.Label(self.master, text="Liczba ruchów: 0", font=("Helvetica", 14))
        #self.move_label.grid(row=3, column=0, pady=5)

        self.misplaced_label = tk.Label(self.master, text="Pomieszane elementy: 0", font=("Helvetica", 14))
        self.misplaced_label.grid(row=4, column=0, pady=5)

    def update_buttons(self):
        # Aktualizujemy przyciski zgodnie ze stanem planszy
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                if value == 0:
                    self.buttons[i][j].config(text="", bg="lightgrey")
                else:
                    self.buttons[i][j].config(text=str(value), bg="SystemButtonFace")
        self.update_counters()

    def update_counters(self):
        # Aktualizacja liczników
        #self.move_label.config(text=f"Liczba ruchów: {self.move_count}")
        misplaced = self.count_misplaced()
        self.misplaced_label.config(text=f"Pomieszane elementy: {misplaced}")
        if misplaced == 5:
            self.shuffle_button.config(state=tk.DISABLED)
        else:
            self.shuffle_button.config(state=tk.NORMAL)

    def count_misplaced(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0 and self.board[i][j] != puzzle.goal_puzzle[i][j]:
                    count += 1
        return count

    def on_tile_click(self, i, j):
        blank_i, blank_j = puzzle.find_zero(self.board)
        if (abs(blank_i - i) == 1 and blank_j == j) or (abs(blank_j - j) == 1 and blank_i == i):
            if i < blank_i:
                move = "U"
            elif i > blank_i:
                move = "D"
            elif j < blank_j:
                move = "L"
            elif j > blank_j:
                move = "R"
            self.board = puzzle.do_the_move(move, self.board)
            self.move_count += 1
            self.update_buttons()

    def shuffle(self):
        if self.count_misplaced() >= 5:
            print("Tasowanie zablokowane – układanka jest prawie ułożona.")
            return
        moves = 5
        for _ in range(moves):
            possible_moves = puzzle.get_possible_moves(self.board)
            if possible_moves:
                move = random.choice(possible_moves)
                self.board = puzzle.do_the_move(move, self.board)
        self.move_count = 0
        self.update_buttons()

    def solve(self):
        solution = puzzle.bfs(self.board)
        if solution:
            self.board = solution
            self.update_buttons()
        else:
            print("BFS nie znalazł rozwiązania. Układanka mogła być pomieszana więcej niż 5 ruchami.")

