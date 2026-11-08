"""
Entry point proyek.

    python main.py play      --difficulty beginner --seed 1 [--plot]
    python main.py solve     --board examples/sample_board.txt [--plot]
    python main.py benchmark --games 5 --seed 0 [--no-plot] [--neural models/x.npz ...]
    python main.py train     --source dataset --data minesweeper_dataset_terbaru.xlsx \
                             --out models/mlp_dataset.npz
    python main.py train     --source selfplay --games 3000 --out models/mlp_selfplay.npz
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
    extra = {}
    if args.neural:
        from src.neural_learner import NeuralWindowSolver
        for path in args.neural:
            label = "MLP (" + path.replace("\\", "/").split("/")[-1].replace(".npz", "") + ")"
            extra[label] = NeuralWindowSolver.load(path)
    summary = ev.run_benchmark(args.games, seed=args.seed, extra_solvers=extra)
    print(summary.to_string(index=False))
    if not args.no_plot:
        ev.plot_benchmark_results(summary)
    return 0


def cmd_train(args) -> int:
    from src.neural_learner import train_from_dataset, train_from_selfplay
    hidden = tuple(int(h) for h in args.hidden.split(","))
    if args.source == "dataset":
        if not args.data:
            print("--data wajib diisi untuk --source dataset", file=sys.stderr)
            return 2
        model, rep = train_from_dataset(
            args.data, nrows=args.rows, hidden=hidden, onehot=not args.raw,
            use_density=args.density, epochs=args.epochs, lr=args.lr,
            loss=args.loss, seed=args.seed)
        m = rep["val_metrics"]
        print(f"Validasi: akurasi={m['accuracy']:.4f} (baseline mayoritas "
              f"{rep['majority_baseline_accuracy']:.4f}) | AUC={m['auc']:.4f} | "
              f"logloss={m['logloss']:.4f}")
    else:
        rows, cols, mines = PRESETS[args.difficulty]
        model, rep = train_from_selfplay(
            n_games=args.games, rows=rows, cols=cols, mines=mines, hidden=hidden,
            onehot=not args.raw, use_density=args.density, epochs=args.epochs,
            lr=args.lr, loss=args.loss, seed=args.seed)
        m = rep["test_metrics"]
        print(f"Uji (game terpisah): akurasi={m['accuracy']:.4f} | "
              f"AUC={m['auc']:.4f} | logloss={m['logloss']:.4f}")
    model.save(args.out)
    print(f"Model disimpan ke {args.out}")
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
    p.add_argument("--neural", nargs="*", default=[], metavar="MODEL.npz",
                   help="ikutkan pelajar MLP (satu atau lebih file model)")
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("train", help="latih MLP jendela 5x5 (jurnal Bab 5.5)")
    p.add_argument("--source", choices=("dataset", "selfplay"), default="dataset",
                   help="dataset = file Kaggle; selfplay = data dibangkitkan saat bermain")
    p.add_argument("--data", help="file dataset (.csv / .xlsx), untuk --source dataset")
    p.add_argument("--rows", type=int, default=None,
                   help="batasi jumlah baris dibaca (default: semua). Awas: baris dataset "
                        "tidak diacak, N baris pertama memberi sampel yang bias")
    p.add_argument("--games", type=int, default=3000, help="jumlah game self-play")
    p.add_argument("--difficulty", choices=PRESETS, default="beginner",
                   help="ukuran papan untuk self-play")
    p.add_argument("--hidden", default="70,80,60,35", help="lapisan tersembunyi (MLP learner3)")
    p.add_argument("--raw", action="store_true", help="24 input mentah (bukan one-hot)")
    p.add_argument("--density", action="store_true", help="tambahkan global_density sebagai fitur")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--loss", choices=("xent", "mse"), default="xent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/mlp_window.npz")
    p.set_defaults(fn=cmd_train)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
