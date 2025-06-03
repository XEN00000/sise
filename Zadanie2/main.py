import tkinter as tk

from createNetwork import MLP_GUI

if __name__ == "__main__":
    root = tk.Tk()
    app = MLP_GUI(root)
    root.mainloop()
