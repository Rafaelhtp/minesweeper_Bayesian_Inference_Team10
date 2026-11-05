"""
Entry point proyek.

    python main.py play      --difficulty beginner --seed 1 [--plot]
    python main.py solve     --board examples/sample_board.txt [--plot]
    python main.py benchmark --games 5 --seed 0 [--no-plot]
"""

import argparse
import sys

import numpy as np

from src.bayesian_solver import BayesianMinesweeperSolver
from src.minesweeper_app import (MinesweeperGame, parse_board_text,
                                 plot_board_and_probabilities, run_cli)

PRESETS = {
    "beginner": (9, 9, 10),
    "intermediate": (16, 16, 40),
    "expert": (16, 30, 99),
}


def cmd_play(args) -> int:
    rows, cols, mines = PRESETS[args.difficulty]
    game = MinesweeperGame(rows, cols, mines)
    game.load_game(rows // 2, cols // 2, rng=np.random.default_rng(args.seed))
    run_cli(game, max_steps=rows * cols * 2)
    if args.plot:
        plot_board_and_probabilities(game)
    return 0 if game.game_status == "WON" else 1


def cmd_solve(args) -> int:
    with open(args.board, encoding="utf-8") as f:
        state, _truth, total = parse_board_text(f.read())
    solver = BayesianMinesweeperSolver()
    r, c, p, pm, reason = solver.recommend_move(state, total)
    print(f"Board {state.shape[0]}x{state.shape[1]}, total bom = {total}")
    print(f"Rekomendasi: baris {r}, kolom {c} | P(bom) = {p * 100:.2f}% | {reason}")
    if args.plot:
        game = MinesweeperGame(*state.shape, total)
        game.load_custom_board(None, state, num_mines=total)
        plot_board_and_probabilities(game, pm, (r, c))
    return 0


def cmd_benchmark(args) -> int:
    from src.data_and_testing import BenchmarkEvaluator
    ev = BenchmarkEvaluator()
    summary = ev.run_benchmark(args.games, seed=args.seed)
    print(summary.to_string(index=False))
    if not args.no_plot:
        ev.plot_benchmark_results(summary)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Minesweeper Bayesian solver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="solver memainkan satu game penuh")
    p.add_argument("--difficulty", choices=PRESETS, default="beginner")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--plot", action="store_true")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("solve", help="rekomendasi langkah untuk board dari file")
    p.add_argument("--board", required=True)
    p.add_argument("--plot", action="store_true")
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("benchmark", help="Baseline vs Bayesian")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(fn=cmd_benchmark)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
