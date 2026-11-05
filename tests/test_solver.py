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

from src.bayesian_solver import BayesianMinesweeperSolver, COVERED, FLAGGED  # noqa: E402
from src.minesweeper_app import MinesweeperGame, parse_board_text  # noqa: E402


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
        gs = np.array([[1, -2, -2],
                       [1, -2, -2],
                       [1, -2, -2]])
        # clue (0,0)=1 -> mine di (0,1) atau (1,1); tidak deterministik
        safe, mines = BayesianMinesweeperSolver().apply_single_point_strategy(gs)
        self.assertEqual((safe, mines), (set(), set()))

        gs = np.array([[1, -2], [-2, -2]])
        gs[0, 1] = FLAGGED          # clue terpenuhi oleh flag -> sisanya aman
        safe, mines = BayesianMinesweeperSolver().apply_single_point_strategy(gs)
        self.assertEqual(safe, {(1, 0), (1, 1)})
        self.assertEqual(mines, set())

        gs = np.array([[2, -2], [-2, 0]])
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
        gs = np.array([[-2, 2, -2],
                       [-2, -2, -2],
                       [-2, -2, -2]])
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


if __name__ == "__main__":
    unittest.main()
