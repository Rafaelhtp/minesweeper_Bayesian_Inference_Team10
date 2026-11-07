"""Contoh: minta rekomendasi langkah untuk board manual.

Jalankan dari root proyek:  python examples/example_manual_board.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bayesian_solver import BayesianMinesweeperSolver  # noqa: E402
from src.minesweeper_app import parse_board_text  # noqa: E402

path = os.path.join(os.path.dirname(__file__), "sample_board.txt")
with open(path, encoding="utf-8") as f:
    state, _truth, total_mines = parse_board_text(f.read())

solver = BayesianMinesweeperSolver()
r, c, p, prob, reason = solver.recommend_move(state, total_mines)
print(f"Langkah terbaik: ({r}, {c})  P(bom)={p:.1%}  [{reason}]")