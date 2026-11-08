"""
bayesian_solver.py
==================

Standalone implementation of a Minesweeper AI solver using a hybrid of
Deterministic Logic (Single Point Strategy / SPS) and Bayesian Inference.

Encoding Convention (disamakan dengan dataset Kaggle minesweeper_dataset)
------------------------------------------------------------------------
- game_state (np.ndarray, dtype=int):
    *  0..8 : uncovered cell showing adjacent mine count
               (pada dataset hanya muncul 0..6).
    *   -1  : covered (unknown) cell            -> sama dengan dataset.
    *   -2  : outside the board (di luar papan) -> sama dengan dataset.
               Sel bernilai -2 tidak pernah dianggap tertutup, clue, maupun
               flag, sehingga solver mengabaikannya.
    *   -3  : flagged mine (treated as known mine). Dataset tidak punya
               flag; kode -3 hanya dipakai program dan tidak bentrok dengan
               nilai dataset.

- total_mines (int): total number of mines on the board.

Key Concepts
------------
- Bordering cells : covered cells with at least one uncovered numbered neighbour.
- Non-bordering    : covered cells with no uncovered numbered neighbour (interior).
- Constraint       : uncovered numbered cell adjacent to bordering cells.

Public API
----------
class BayesianMinesweeperSolver:
    - get_bordering_and_constraints(game_state)
    - apply_single_point_strategy(game_state)
    - compute_bayesian_probabilities(game_state, total_mines)
    - recommend_move(game_state, total_mines)

This module depends only on `numpy` and Python's standard library.

Usage
-----
    from src.bayesian_solver import BayesianMinesweeperSolver

    solver = BayesianMinesweeperSolver()
    row, col, p, matrix, reason = solver.recommend_move(game_state, total_mines)

Or run directly for a demo:

    python src/bayesian_solver.py
"""

from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

__all__ = ["BayesianMinesweeperSolver", "COVERED", "OUT_OF_BOARD", "FLAGGED"]

# --------------------------------------------------------------------------- #
# Encoding constants
# --------------------------------------------------------------------------- #
COVERED: int = -1       # sel tertutup            (dataset: -1)
OUT_OF_BOARD: int = -2  # di luar papan           (dataset: -2)
FLAGGED: int = -3       # bendera (hanya program) (dataset: tidak ada)


# Search limits for exact enumeration of one border component.
NODE_BUDGET: int = 1_500_000      # max DFS nodes per component
MAX_COMPONENT_CELLS: int = 400    # also keeps recursion depth safe


class _BudgetExceeded(Exception):
    """Raised when a component is too large to enumerate exactly."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _neighbours(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """8-connectivity neighbours, only valid coordinates."""
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out.append((nr, nc))
    return out


# --------------------------------------------------------------------------- #
# Main solver class
# --------------------------------------------------------------------------- #
class BayesianMinesweeperSolver:
    """Hybrid SPS + Bayesian Inference Minesweeper solver."""

    # ------------------------------------------------------------------ #
    # 1. Frontier extraction
    # ------------------------------------------------------------------ #
    def get_bordering_and_constraints(
        self, game_state: np.ndarray
    ) -> Tuple[Set[Tuple[int, int]],
               Set[Tuple[int, int]],
               Set[Tuple[int, int]]]:
        """
        Extract frontier information.

        Returns
        -------
        bordering   : covered cells (-1) adjacent to at least one clue (1..8)
        constraints : clue cells (1..8) adjacent to at least one covered cell
        non_border  : covered cells (-1) with no adjacent clue (interior cells)
        """
        rows, cols = game_state.shape
        bordering: Set[Tuple[int, int]] = set()
        constraints: Set[Tuple[int, int]] = set()

        for r in range(rows):
            for c in range(cols):
                if 1 <= game_state[r, c] <= 8:
                    constraints.add((r, c))
                    for nr, nc in _neighbours(r, c, rows, cols):
                        if game_state[nr, nc] == COVERED:
                            bordering.add((nr, nc))

        non_border: Set[Tuple[int, int]] = set()
        for r in range(rows):
            for c in range(cols):
                if game_state[r, c] == COVERED and (r, c) not in bordering:
                    non_border.add((r, c))

        return bordering, constraints, non_border

    # ------------------------------------------------------------------ #
    # 2. Single Point Strategy (deterministic logic)
    # ------------------------------------------------------------------ #
    def apply_single_point_strategy(
        self, game_state: np.ndarray
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Single Point Strategy.

        For every uncovered clue cell x:
            mines_left      = value(x) - flagged_neighbours(x)
            covered_neighs  = list of covered neighbours
            - if mines_left == 0            -> all covered neighbours SAFE (P=0)
            - if mines_left == len(covered) -> all covered neighbours MINES (P=1)

        Returns
        -------
        safe_cells : guaranteed safe covered cells
        mine_cells : guaranteed mine covered cells
        """
        rows, cols = game_state.shape
        safe: Set[Tuple[int, int]] = set()
        mines: Set[Tuple[int, int]] = set()

        for r in range(rows):
            for c in range(cols):
                val = int(game_state[r, c])
                if not (0 <= val <= 8):
                    continue

                covered_nb: List[Tuple[int, int]] = []
                flagged_nb = 0
                for nr, nc in _neighbours(r, c, rows, cols):
                    if game_state[nr, nc] == COVERED:
                        covered_nb.append((nr, nc))
                    elif game_state[nr, nc] == FLAGGED:
                        flagged_nb += 1

                if not covered_nb:
                    continue

                mines_left = val - flagged_nb
                if mines_left == 0:
                    safe.update(covered_nb)
                elif mines_left == len(covered_nb):
                    mines.update(covered_nb)

        return safe, mines

    # ------------------------------------------------------------------ #
    # 3. Component splitting (independent subproblems)
    # ------------------------------------------------------------------ #
    def _split_components(
        self,
        bordering: Set[Tuple[int, int]],
        constraints: Set[Tuple[int, int]],
        rows: int,
        cols: int,
    ) -> List[Dict]:
        """
        Partition (bordering, constraints) into connected components
        via BFS on the bipartite constraint<->border graph.
        """
        cons_to_border = defaultdict(list)
        border_to_cons = defaultdict(list)
        for (cr, cc) in constraints:
            for nr, nc in _neighbours(cr, cc, rows, cols):
                if (nr, nc) in bordering:
                    cons_to_border[(cr, cc)].append((nr, nc))
                    border_to_cons[(nr, nc)].append((cr, cc))

        visited: Set[Tuple[int, int]] = set()
        components: List[Dict] = []

        for start in bordering:
            if start in visited:
                continue

            comp_border: Set[Tuple[int, int]] = set()
            comp_cons: Set[Tuple[int, int]] = set()
            stack: List[Tuple[str, Tuple[int, int]]] = [("B", start)]

            while stack:
                kind, node = stack.pop()
                if kind == "B":
                    if node in comp_border:
                        continue
                    comp_border.add(node)
                    visited.add(node)
                    for cons in border_to_cons.get(node, []):
                        if cons not in comp_cons:
                            stack.append(("C", cons))
                else:  # constraint node
                    if node in comp_cons:
                        continue
                    comp_cons.add(node)
                    for nb in cons_to_border.get(node, []):
                        if nb not in comp_border:
                            stack.append(("B", nb))

            components.append({
                "border": sorted(comp_border),
                "constraints": sorted(comp_cons),
            })

        return components

    # ------------------------------------------------------------------ #
    # 4. Enumerate valid mine assignments per component
    # ------------------------------------------------------------------ #
    def _enumerate_component(
        self,
        comp: Dict,
        work: np.ndarray,
        rows: int,
        cols: int,
        node_budget: int = NODE_BUDGET,
    ) -> Dict[int, Tuple[int, List[int]]]:
        """
        Backtracking with incremental forward checking over one component.

        Instead of storing every assignment (memory blow-up), we only keep,
        for each total mine count k:
            * how many valid assignments use exactly k mines, and
            * for every border cell, in how many of those it is a mine.

        Returns
        -------
        {k: (n_solutions, per_cell_mine_counts)}
            Empty dict  -> constraints are contradictory (infeasible).
        Raises _BudgetExceeded if the search tree is too large.
        """
        border_cells: List[Tuple[int, int]] = comp["border"]
        idx = {cell: i for i, cell in enumerate(border_cells)}
        n = len(border_cells)
        if n > MAX_COMPONENT_CELLS:
            raise _BudgetExceeded

        need: List[int] = []
        members: List[List[int]] = []
        for (cr, cc) in comp["constraints"]:
            flagged = 0
            bidx: List[int] = []
            for nr, nc in _neighbours(cr, cc, rows, cols):
                if work[nr, nc] == FLAGGED:
                    flagged += 1
                elif (nr, nc) in idx:
                    bidx.append(idx[(nr, nc)])
            k = int(work[cr, cc]) - flagged
            if k < 0 or k > len(bidx):
                return {}                       # contradictory clue
            need.append(k)
            members.append(bidx)

        cell_cons: List[List[int]] = [[] for _ in range(n)]
        for ci, bidx in enumerate(members):
            for j in bidx:
                cell_cons[j].append(ci)
        free = [len(m) for m in members]

        results: Dict[int, List] = {}
        chosen: List[int] = []
        nodes = 0

        def dfs(i: int) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > node_budget:
                raise _BudgetExceeded
            if i == n:
                ent = results.get(len(chosen))
                if ent is None:
                    ent = [0, [0] * n]
                    results[len(chosen)] = ent
                ent[0] += 1
                cnt = ent[1]
                for j in chosen:
                    cnt[j] += 1
                return
            cons = cell_cons[i]
            for v in (0, 1):
                ok = True
                for ci in cons:
                    need[ci] -= v
                    free[ci] -= 1
                for ci in cons:
                    if need[ci] < 0 or need[ci] > free[ci]:
                        ok = False
                        break
                if ok:
                    if v:
                        chosen.append(i)
                    dfs(i + 1)
                    if v:
                        chosen.pop()
                for ci in cons:
                    need[ci] += v
                    free[ci] += 1

        dfs(0)
        return {k: (v[0], v[1]) for k, v in results.items()}

    def _local_estimate(
        self, comp: Dict, work: np.ndarray, rows: int, cols: int
    ) -> Dict[Tuple[int, int], float]:
        """
        Cheap fallback for components too large to enumerate: average of
        (mines_needed / unknown_neighbours) over the clues touching each cell.
        """
        border = set(comp["border"])
        sums: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        for (cr, cc) in comp["constraints"]:
            flagged, unk = 0, []
            for nr, nc in _neighbours(cr, cc, rows, cols):
                if work[nr, nc] == FLAGGED:
                    flagged += 1
                elif (nr, nc) in border:
                    unk.append((nr, nc))
            if not unk:
                continue
            p = min(max((int(work[cr, cc]) - flagged) / len(unk), 0.0), 1.0)
            for cell in unk:
                sums[cell].append(p)
        return {cell: float(np.mean(v)) for cell, v in sums.items()}

    # ------------------------------------------------------------------ #
    # 5. Bayesian posterior probabilities
    # ------------------------------------------------------------------ #
    def compute_bayesian_probabilities(
        self,
        game_state: np.ndarray,
        total_mines: int,
    ) -> Dict[Tuple[int, int], float]:
        """
        Exact P(mine | evidence) for every covered cell.

        Every border configuration C is weighted by how many ways the rest of
        the mines can be placed in the interior (non-border) cells:

            P(C | E)  ∝  C(n_interior, remaining_mines - mines(C))

        Components are enumerated independently (counts per mine total) and
        merged by convolution, so no joint product over components is needed.
        All arithmetic uses Python ints -> exact, no overflow/underflow.
        """
        rows, cols = game_state.shape
        prob: Dict[Tuple[int, int], float] = {}

        # (a) SPS first — deterministic 0/1
        safe_sps, mine_sps = self.apply_single_point_strategy(game_state)
        for cell in safe_sps:
            prob[cell] = 0.0
        for cell in mine_sps:
            prob[cell] = 1.0

        # Work on a copy where SPS-proven mines count as flags. Otherwise the
        # constraint solver would not know they exist and would miscount.
        work = game_state.copy()
        for (r, c) in mine_sps:
            work[r, c] = FLAGGED
        flagged = int(np.sum(work == FLAGGED))
        remaining_mines = max(0, total_mines - flagged)

        # (b) Frontier extraction
        bordering, constraints, non_border = \
            self.get_bordering_and_constraints(work)
        bordering = {c for c in bordering if c not in safe_sps}
        non_border = {c for c in non_border if c not in safe_sps}
        n_non_border = len(non_border)

        # (c) No frontier -> pure interior prior
        if not bordering:
            if n_non_border > 0:
                p = min(max(remaining_mines / n_non_border, 0.0), 1.0)
                for cell in non_border:
                    prob[cell] = p
            return prob

        # (d) Enumerate each independent component
        exact: List[Tuple[Dict, Dict[int, Tuple[int, List[int]]]]] = []
        approx_expected = 0.0
        for comp in self._split_components(bordering, constraints, rows, cols):
            try:
                res = self._enumerate_component(comp, work, rows, cols)
            except _BudgetExceeded:
                est = self._local_estimate(comp, work, rows, cols)
                for cell in comp["border"]:
                    prob[cell] = est.get(cell, 0.5)
                    approx_expected += prob[cell]
                continue
            if not res:                       # contradictory evidence
                for cell in comp["border"]:
                    prob[cell] = 0.5
                continue
            exact.append((comp, res))

        # (e) Merge components + interior prior (exact big-int arithmetic)
        R = max(0, remaining_mines - int(round(approx_expected)))
        N = n_non_border

        def tail(m: int, use_prior: bool) -> int:
            if not use_prior:
                return 1
            k = R - m
            return comb(N, k) if 0 <= k <= N else 0

        dists = [{k: v[0] for k, v in res.items()} for _, res in exact]

        def convolve(ds: List[Dict[int, int]]) -> Dict[int, int]:
            out: Dict[int, int] = {0: 1}
            for d in ds:
                nxt: Dict[int, int] = defaultdict(int)
                for k1, v1 in out.items():
                    for k2, v2 in d.items():
                        nxt[k1 + k2] += v1 * v2
                out = nxt
            return out

        full = convolve(dists)
        use_prior = True
        Z = sum(v * tail(m, True) for m, v in full.items())
        if Z == 0:
            # total_mines inconsistent with the evidence -> ignore global count
            use_prior = False
            Z = sum(full.values())

        for i, (comp, res) in enumerate(exact):
            others = convolve(dists[:i] + dists[i + 1:])
            w = {k: sum(v * tail(k + m, use_prior) for m, v in others.items())
                 for k in res}
            for j, cell in enumerate(comp["border"]):
                s = sum(res[k][1][j] * w[k] for k in res)
                prob[cell] = float(s / Z) if Z else 0.5

        # (f) Interior cells share one probability
        if N > 0:
            if exact and use_prior and Z:
                e_int = sum(v * tail(m, True) * (R - m)
                            for m, v in full.items())
                p_bg = float(e_int / Z) / N
            else:
                border_mines = sum(prob[c] for c in bordering if c in prob)
                p_bg = (R - border_mines) / N
            p_bg = float(min(max(p_bg, 0.0), 1.0))
            for cell in non_border:
                prob[cell] = p_bg

        # (g) Safety net: unclassified covered cells
        total_covered = int(np.sum(game_state == COVERED))
        fallback = float(min(max(remaining_mines / max(1, total_covered),
                                 0.0), 1.0))
        for r in range(rows):
            for c in range(cols):
                if game_state[r, c] == COVERED and (r, c) not in prob:
                    prob[(r, c)] = fallback

        return prob

    # ------------------------------------------------------------------ #
    # 6. Move recommendation
    # ------------------------------------------------------------------ #
    def recommend_move(
        self,
        game_state: np.ndarray,
        total_mines: int,
    ) -> Tuple[int, int, float, np.ndarray, str]:
        """
        Recommend the safest next move.

        Returns
        -------
        best_row, best_col : int
        min_prob           : float
        prob_matrix        : np.ndarray  (rows x cols; np.nan for revealed)
        reasoning_type     : "SPS_DETERMINISTIC" | "BAYESIAN_BORDER" |
                             "BAYESIAN_PRIOR_GUESS"
        """
        rows, cols = game_state.shape
        prob_matrix = np.full((rows, cols), np.nan, dtype=float)

        # (1) SPS first — deterministic
        safe_sps, _mine_sps = self.apply_single_point_strategy(game_state)
        if safe_sps:
            best = self._corner_tiebreak(safe_sps, rows, cols)
            try:
                full = self.compute_bayesian_probabilities(
                    game_state, total_mines
                )
                for (r, c), p in full.items():
                    prob_matrix[r, c] = p
            except Exception:
                pass
            prob_matrix[best[0], best[1]] = 0.0
            return best[0], best[1], 0.0, prob_matrix, "SPS_DETERMINISTIC"

        # (2) Bayesian full distribution
        full = self.compute_bayesian_probabilities(game_state, total_mines)

        candidates: List[Tuple[Tuple[int, int], float]] = []
        for (r, c), p in full.items():
            if game_state[r, c] == COVERED:
                candidates.append(((r, c), float(p)))
                prob_matrix[r, c] = float(p)

        if not candidates:
            return -1, -1, 1.0, prob_matrix, "BAYESIAN_PRIOR_GUESS"

        # (3) Pick minimum, tie-break corner > edge > interior
        candidates.sort(
            key=lambda x: (round(x[1], 9), self._tiebreak_key(x[0], rows, cols))
        )
        (br, bc), best_p = candidates[0]

        bordering, _cons, _nb = self.get_bordering_and_constraints(game_state)
        if (br, bc) in bordering:
            reasoning = "BAYESIAN_BORDER"
        else:
            reasoning = "BAYESIAN_PRIOR_GUESS"

        return br, bc, float(best_p), prob_matrix, reasoning

    # ---- small helpers ---------------------------------------------- #
    @staticmethod
    def _tiebreak_key(
        cell: Tuple[int, int], rows: int, cols: int
    ) -> Tuple[int, int, int]:
        r, c = cell
        is_corner = (r in (0, rows - 1)) and (c in (0, cols - 1))
        is_edge = (r in (0, rows - 1)) or (c in (0, cols - 1))
        rank = 0 if is_corner else (1 if is_edge else 2)
        return (rank, r, c)

    @staticmethod
    def _corner_tiebreak(
        cells: Set[Tuple[int, int]], rows: int, cols: int
    ) -> Tuple[int, int]:
        return sorted(
            cells,
            key=lambda rc: BayesianMinesweeperSolver._tiebreak_key(
                rc, rows, cols
            ),
        )[0]


# --------------------------------------------------------------------------- #
# Demo (standalone)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Partially revealed 9x9 Beginner board.
    #   -1  = covered
    #   0..8 = revealed clue
    sample = np.array([
        [ 0,  0,  1, -1, -1, -1, -1, -1, -1],
        [ 0,  0,  1, -1, -1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1, -1, -1, -1, -1],
        [ 1,  2,  2, -1, -1, -1, -1, -1, -1],
        [ 1,  2, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  2, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ], dtype=int)
    total_mines = 10  # Beginner preset

    solver = BayesianMinesweeperSolver()

    # 1) SPS demonstration
    safe, mines = solver.apply_single_point_strategy(sample)
    print("=" * 68)
    print("SINGLE POINT STRATEGY (Deterministic)")
    print("=" * 68)
    print(f"  Guaranteed safe cells : {sorted(safe) if safe else 'none'}")
    print(f"  Guaranteed mine cells : {sorted(mines) if mines else 'none'}")

    # 2) Full Bayesian posterior
    prob_map = solver.compute_bayesian_probabilities(sample, total_mines)
    prob_matrix = np.full(sample.shape, np.nan, dtype=float)
    for (r, c), p in prob_map.items():
        prob_matrix[r, c] = p

    print()
    print("=" * 68)
    print("BAYESIAN POSTERIOR P(MINE | EVIDENCE)")
    print("=" * 68)
    print("      " + "".join(f"{c:>8}" for c in range(sample.shape[1])))
    for r in range(sample.shape[0]):
        row_str = f"  {r:>2}  "
        for c in range(sample.shape[1]):
            if sample[r, c] != COVERED:
                row_str += f"{'  --':>8}"
            else:
                row_str += f"{prob_matrix[r, c] * 100:>7.2f}%"
        print(row_str)

    # 3) Recommendation
    br, bc, bp, _pm, reasoning = solver.recommend_move(
        sample, total_mines
    )
    print()
    print("=" * 68)
    print("RECOMMENDED NEXT MOVE")
    print("=" * 68)
    print(f"  Cell       : ({br}, {bc})")
    print(f"  P(mine)    : {bp * 100:.2f}%")
    print(f"  Reasoning  : {reasoning}")
    print("=" * 68)