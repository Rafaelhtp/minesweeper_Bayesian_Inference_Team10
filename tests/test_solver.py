"""
Unit test untuk solver & engine.

Jalankan dari root proyek:
    python -m unittest discover -s tests -v
(atau `pytest` jika terpasang)
"""

import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bayesian_solver import (BayesianMinesweeperSolver, COVERED, FLAGGED,  # noqa: E402
                                 OUT_OF_BOARD)
from src.minesweeper_app import MINE, MinesweeperGame, parse_board_text  # noqa: E402
from src.data_and_testing import KAGGLE_CELL_COLUMNS, DatasetLoader  # noqa: E402
from src.neural_learner import (NeuralWindowSolver, NumpyMLP, WindowMLP,  # noqa: E402
                                generate_selfplay_windows, train_from_dataset,
                                window_cells)

C = COVERED  # singkatan agar papan di test mudah dibaca


def brute_force_probs(state, total_mines):
    """Marginal P(mine) eksak dengan mengenumerasi SEMUA penempatan bom."""
    rows, cols = state.shape
    covered = [(r, c) for r in range(rows) for c in range(cols)
               if state[r, c] == COVERED]
    flagged = int(np.sum(state == FLAGGED))
    k = total_mines - flagged
    clues = [(r, c) for r in range(rows) for c in range(cols)
             if 0 <= state[r, c] <= 8]
    counts = {cell: 0 for cell in covered}
    valid = 0
    for combo in itertools.combinations(range(len(covered)), k):
        mines = {covered[i] for i in combo}
        ok = True
        for (r, c) in clues:
            n = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols:
                        rr, cc = r + dr, c + dc
                        if state[rr, cc] == FLAGGED or (rr, cc) in mines:
                            n += 1
            if n != state[r, c]:
                ok = False
                break
        if ok:
            valid += 1
            for cell in mines:
                counts[cell] += 1
    return {cell: counts[cell] / valid for cell in covered} if valid else None


class TestSPS(unittest.TestCase):
    def test_safe_and_mine_detection(self):
        gs = np.array([[1, C, C],
                       [1, C, C],
                       [1, C, C]])
        # clue (0,0)=1 -> mine di (0,1) atau (1,1); tidak deterministik
        safe, mines = BayesianMinesweeperSolver().apply_single_point_strategy(gs)
        self.assertEqual((safe, mines), (set(), set()))

        gs = np.array([[1, C], [C, C]])
        gs[0, 1] = FLAGGED          # clue terpenuhi oleh flag -> sisanya aman
        safe, mines = BayesianMinesweeperSolver().apply_single_point_strategy(gs)
        self.assertEqual(safe, {(1, 0), (1, 1)})
        self.assertEqual(mines, set())

        gs = np.array([[2, C], [C, 0]])
        safe, mines = BayesianMinesweeperSolver().apply_single_point_strategy(gs)
        self.assertEqual(mines, {(0, 1), (1, 0)})


class TestBayesExact(unittest.TestCase):
    def test_matches_brute_force_on_random_boards(self):
        rng = np.random.default_rng(7)
        solver = BayesianMinesweeperSolver()
        checked = 0
        for _ in range(150):
            rows, cols = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            n_mines = int(rng.integers(1, max(2, rows * cols // 4)))
            g = MinesweeperGame(rows, cols, n_mines)
            g.load_game(int(rng.integers(rows)), int(rng.integers(cols)), rng=rng)
            # buka beberapa sel aman acak agar papan lebih kaya
            for _ in range(int(rng.integers(0, 4))):
                safe = [(r, c) for r in range(rows) for c in range(cols)
                        if g.game_state[r, c] == COVERED and g.board_values[r, c] >= 0]
                if safe and g.game_status == "PLAYING":
                    g.uncover(*safe[int(rng.integers(len(safe)))])
            if g.count_covered() > 18 or g.count_covered() == 0:
                continue
            expected = brute_force_probs(g.game_state, g.num_mines)
            got = solver.compute_bayesian_probabilities(g.game_state, g.num_mines)
            for cell, p in expected.items():
                self.assertAlmostEqual(got[cell], p, places=9,
                                       msg=f"{cell}\n{g.game_state}")
            checked += 1
        self.assertGreater(checked, 60)

    def test_sps_mine_is_counted(self):
        """Regresi: bom hasil SPS (belum diflag) harus ikut dihitung."""
        gs = np.array([[C, 2, C],
                       [C, C, C],
                       [C, C, C]])
        # clue 2 di (0,1): tetangga covered = (0,0),(0,2),(1,0),(1,1),(1,2)
        p = BayesianMinesweeperSolver().compute_bayesian_probabilities(gs, 2)
        self.assertAlmostEqual(sum(p[c] for c in [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]), 2.0)

    def test_large_component_is_not_all_zero(self):
        """Regresi: komponen >32 sel dulu jatuh ke sampling acak -> p=0 palsu."""
        g = MinesweeperGame(16, 30, 99)
        g.load_game(8, 15, rng=np.random.default_rng(5))
        p = BayesianMinesweeperSolver().compute_bayesian_probabilities(
            g.game_state, g.num_mines)
        border_total = sum(v for v in p.values())
        # total bom yang diharapkan harus ~ jumlah bom tersisa
        self.assertAlmostEqual(border_total, g.num_mines, delta=0.5)


class TestEngineAndParser(unittest.TestCase):
    def test_first_click_safe(self):
        for seed in range(20):
            g = MinesweeperGame(9, 9, 10)
            g.load_game(4, 4, rng=np.random.default_rng(seed))
            self.assertEqual(g.game_status, "PLAYING")

    def test_parse_with_mines_and_count(self):
        gs, bv, tm = parse_board_text("# mines=2\n? 1 ?\n? ? ?")
        self.assertIsNone(bv)
        self.assertEqual(tm, 2)
        self.assertEqual(gs.shape, (2, 3))

    def test_parse_ground_truth(self):
        gs, bv, tm = parse_board_text("M ? ?\n? ? ?")
        self.assertEqual(tm, 1)
        self.assertEqual(bv[0, 1], 1)
        self.assertEqual(gs[0, 0], COVERED)

    def test_solver_wins_easy_games(self):
        solver = BayesianMinesweeperSolver()
        wins = 0
        for seed in range(10):
            g = MinesweeperGame(9, 9, 10)
            g.load_game(4, 4, rng=np.random.default_rng(seed))
            while g.game_status == "PLAYING":
                r, c, *_ = solver.recommend_move(g.game_state, g.num_mines)
                if r < 0:
                    break
                g.uncover(r, c)
            wins += g.game_status == "WON"
        self.assertGreaterEqual(wins, 5)


class TestDatasetEncoding(unittest.TestCase):
    """Encoding program harus identik dengan dataset Kaggle."""

    def test_constants_match_dataset(self):
        self.assertEqual(COVERED, -1)        # dataset: tertutup
        self.assertEqual(OUT_OF_BOARD, -2)   # dataset: di luar papan
        self.assertEqual(FLAGGED, -3)        # hanya program
        self.assertNotIn(MINE, (-1, -2, -3)) # kode bom internal tidak bentrok

    def test_parser_uses_dataset_codes(self):
        gs, bv, tm = parse_board_text("-1 1 -2\nF 0 -1\n# mines=2")
        self.assertEqual(gs.tolist(), [[-1, 1, -2], [-3, 0, -1]])
        self.assertIsNone(bv)

    def test_new_game_starts_all_covered_as_minus_one(self):
        g = MinesweeperGame(3, 3, 1)
        self.assertTrue(np.all(g.game_state == -1))

    def test_exploded_mine_is_distinct_from_covered(self):
        g = MinesweeperGame(3, 3, 1)
        g.load_custom_board(np.array([[MINE, 1, 0], [1, 1, 0], [0, 0, 0]]))
        g.uncover(0, 0)
        self.assertEqual(g.game_status, "LOST")
        self.assertEqual(g.game_state[0, 0], MINE)
        self.assertNotEqual(g.game_state[0, 0], COVERED)

    def test_out_of_board_cells_are_ignored_by_solver(self):
        # jendela 5x5 gaya dataset di sudut papan: baris/kolom atas-kiri = -2
        O = OUT_OF_BOARD
        win = np.array([[O, O, O, O, O],
                        [O, O, O, O, O],
                        [O, O, C, 1, C],
                        [O, O, 1, 1, C],
                        [O, O, C, C, C]])
        solver = BayesianMinesweeperSolver()
        bordering, constraints, non_border = solver.get_bordering_and_constraints(win)
        for r, c in bordering | non_border:
            self.assertEqual(win[r, c], COVERED)
        prob = solver.compute_bayesian_probabilities(win, 3)
        self.assertTrue(all(win[r, c] == COVERED for r, c in prob))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in prob.values()))

    def test_kaggle_loader_builds_5x5_windows(self):
        import pandas as pd
        row = {c: -1 for c in KAGGLE_CELL_COLUMNS}
        row["cell_-2_-2"] = -2     # pojok kiri atas = di luar papan
        row["cell_0_1"] = 3        # tepat di kanan target
        row["cell_2_2"] = 0        # pojok kanan bawah
        row.update({"global_density": 0.2, "safe": 1})
        item = DatasetLoader.from_kaggle_dataset(pd.DataFrame([row]))[0]
        gs = item["game_state"]
        self.assertEqual(gs.shape, (5, 5))
        self.assertEqual(gs[0, 0], OUT_OF_BOARD)
        self.assertEqual(gs[2, 3], 3)
        self.assertEqual(gs[4, 4], 0)
        self.assertEqual(gs[2, 2], COVERED)   # sel target diisi tertutup
        self.assertEqual(item["safe"], 1)
        self.assertGreaterEqual(item["total_mines"], 1)

    def test_kaggle_loader_rejects_unknown_codes(self):
        import pandas as pd
        row = {c: -1 for c in KAGGLE_CELL_COLUMNS}
        row["cell_0_1"] = 9
        with self.assertRaises(ValueError):
            DatasetLoader.from_kaggle_dataset(pd.DataFrame([row]))


class TestJournalAlignment(unittest.TestCase):
    """Kesesuaian dengan jurnal arXiv:2212.10446 (klik pertama, MLP jendela 5x5)."""

    def test_first_click_safe_is_default_and_may_be_nonzero(self):
        nonzero = 0
        for seed in range(80):
            g = MinesweeperGame(9, 9, 10)
            g.load_game(4, 4, rng=np.random.default_rng(seed))
            self.assertEqual(g.game_status, "PLAYING")        # tidak pernah bom
            nonzero += int(g.game_state[4, 4] > 0)
        self.assertGreater(nonzero, 0)   # bukan "first click zero"

    def test_first_click_zero_option(self):
        for seed in range(30):
            g = MinesweeperGame(9, 9, 10)
            g.load_game(4, 4, rng=np.random.default_rng(seed), first_click="zero")
            self.assertEqual(g.game_state[4, 4], 0)
        with self.assertRaises(ValueError):
            MinesweeperGame(9, 9, 10).load_game(4, 4, first_click="lainnya")

    def test_window_order_matches_dataset_columns(self):
        state = (np.arange(49).reshape(7, 7) % 9).astype(int)
        r, c = 3, 4
        win = window_cells(state, [(r, c)])[0]
        for value, name in zip(win, KAGGLE_CELL_COLUMNS):
            _, dr, dc = name.split("_")
            self.assertEqual(value, state[r + int(dr), c + int(dc)])

    def test_window_pads_outside_and_maps_flag_to_covered(self):
        state = np.array([[FLAGGED, 1], [1, 0]])
        win = window_cells(state, [(0, 0)])[0]
        self.assertEqual(int((win == OUT_OF_BOARD).sum()), 21)
        self.assertNotIn(FLAGGED, win.tolist())

    def test_mlp_learns_simple_function(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 2, size=(2000, 2)).astype(np.float32)
        y = (X[:, 0] * X[:, 1]).astype(np.float32)             # AND
        net = NumpyMLP(2, (8, 8), seed=1)
        hist = net.fit(X, y, epochs=30, batch_size=64, lr=0.01, verbose=False)
        self.assertLess(hist[-1]["train_logloss"], hist[0]["train_logloss"])
        self.assertGreater(float(((net.predict(X) > 0.5) == (y > 0.5)).mean()), 0.95)

    def test_mlp_gradients_match_numeric(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 4))
        y = (rng.random(6) > 0.5).astype(np.float64)
        for loss in ("xent", "mse"):
            net = NumpyMLP(4, (3,), seed=2)
            net.W = [w.astype(np.float64) for w in net.W]
            net.b = [b.astype(np.float64) for b in net.b]

            def value():
                p = net._forward(X)[1][:, 0]
                return (-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
                        if loss == "xent" else ((p - y) ** 2).mean())
            gW, _gb, _ = net._grads(X, y, loss)
            for idx in np.ndindex(net.W[0].shape):
                old, h = net.W[0][idx], 1e-6
                net.W[0][idx] = old + h
                up = value()
                net.W[0][idx] = old - h
                down = value()
                net.W[0][idx] = old
                self.assertAlmostEqual((up - down) / (2 * h), gW[0][idx], places=6)

    def test_model_save_load_roundtrip(self):
        import os
        import tempfile
        model = WindowMLP(hidden=(6, 5), onehot=True, use_density=True, seed=3)
        cells = np.random.default_rng(0).integers(-2, 9, size=(10, 24))
        dens = np.full(10, 0.15)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.npz")
            model.save(path)
            again = WindowMLP.load(path)
        np.testing.assert_allclose(model.predict_mine(cells, dens),
                                   again.predict_mine(cells, dens), rtol=1e-6)

    def test_neural_solver_always_returns_covered_cell(self):
        solver = NeuralWindowSolver(WindowMLP(hidden=(8, 8), onehot=True, seed=0))
        for seed in range(5):
            g = MinesweeperGame(9, 9, 10)
            g.load_game(4, 4, rng=np.random.default_rng(seed))
            while g.game_status == "PLAYING":
                r, c, p, pm, reason = solver.recommend_move(g.game_state, 10)
                self.assertEqual(g.game_state[r, c], COVERED)
                self.assertTrue(0.0 <= p <= 1.0)
                g.uncover(r, c)
        all_covered = np.full((5, 5), COVERED)
        r, c, *_ = solver.recommend_move(all_covered, 3)   # tanpa sel perbatasan
        self.assertEqual(all_covered[r, c], COVERED)
        done = np.zeros((3, 3), dtype=int)
        self.assertEqual(solver.recommend_move(done, 1)[:2], (-1, -1))

    def test_selfplay_windows_shape_and_labels(self):
        cells, dens, mine = generate_selfplay_windows(n_games=15, seed=0)
        self.assertEqual(cells.shape[1], 24)
        self.assertEqual(len(cells), len(mine))
        self.assertTrue(set(np.unique(mine)).issubset({0.0, 1.0}))
        self.assertTrue(set(np.unique(cells).tolist()).issubset(set(range(-2, 9))))
        self.assertTrue(0.3 < mine.mean() < 0.7)   # sel perbatasan: sekitar separuh bom

    def test_train_from_dataset_dataframe_dedups_before_split(self):
        import pandas as pd
        cells, dens, mine = generate_selfplay_windows(n_games=20, seed=1)
        df = pd.DataFrame(cells, columns=KAGGLE_CELL_COLUMNS)
        df["global_density"] = dens
        df["safe"] = (1 - mine).astype(int)
        df = pd.concat([df, df.head(200)], ignore_index=True)    # duplikat sengaja
        model, rep = train_from_dataset(df, epochs=1, hidden=(8,), verbose=False)
        self.assertLess(rep["rows_after_dedup"], rep["rows_read"])
        self.assertEqual(rep["n_train"] + rep["n_val"], rep["rows_after_dedup"])
        self.assertIn("auc", rep["val_metrics"])
        self.assertEqual(len(model.predict_mine(cells[:3])), 3)


if __name__ == "__main__":
    unittest.main()
