"""
explain.py
==========
Rincian langkah-langkah perhitungan Bayesian (untuk ditampilkan di layar).

Memakai fungsi internal BayesianMinesweeperSolver (komponen, enumerasi) lalu
menyusun ulang konvolusi + bobot supaya setiap angka bisa ditampilkan:

    P(B_i=1 | E) = SUM_k  n_mine(i,k) * W(k)  /  Z
    W(k)         = C(N_interior, R - k - m_lain)     (prior jumlah bom global)

Hasil `explain()` berupa dict yang aman di-JSON-kan (bilangan besar -> string).
"""
from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .bayesian_solver import (BayesianMinesweeperSolver, COVERED, FLAGGED,
                                  _BudgetExceeded, _neighbours)
except ImportError:  # dijalankan dengan sys.path ke src/
    from bayesian_solver import (BayesianMinesweeperSolver, COVERED, FLAGGED,
                                 _BudgetExceeded, _neighbours)


def _convolve(ds: List[Dict[int, int]]) -> Dict[int, int]:
    out: Dict[int, int] = {0: 1}
    for d in ds:
        nxt: Dict[int, int] = defaultdict(int)
        for k1, v1 in out.items():
            for k2, v2 in d.items():
                nxt[k1 + k2] += v1 * v2
        out = nxt
    return dict(out)


def _fmt(x: int) -> str:
    """Bilangan besar -> teks ringkas (mis. 1.23e+15) agar muat di layar."""
    return str(x) if x < 10 ** 9 else f"{x:.3e}"


def explain(solver: BayesianMinesweeperSolver, game_state: np.ndarray,
            total_mines: int) -> Dict[str, Any]:
    rows, cols = game_state.shape
    safe_sps, mine_sps = solver.apply_single_point_strategy(game_state)

    work = game_state.copy()
    for (r, c) in mine_sps:
        work[r, c] = FLAGGED
    flagged = int(np.sum(work == FLAGGED))
    R = max(0, total_mines - flagged)

    bordering, constraints, non_border = solver.get_bordering_and_constraints(work)
    bordering = {c for c in bordering if c not in safe_sps}
    non_border = {c for c in non_border if c not in safe_sps}
    N = len(non_border)

    info: Dict[str, Any] = {
        "rows": rows, "cols": cols, "total_mines": int(total_mines),
        "flagged": flagged, "R": R, "N": N,
        "n_border": len(bordering),
        "n_sps_safe": len(safe_sps), "n_sps_mine": len(mine_sps),
        "components": [], "mix": [], "Z": "0", "approx": False,
        "cells": {},   # "r,c" -> rincian per sel
    }

    # --- SPS: alasan deterministik per sel --------------------------------
    sps_reason: Dict[Tuple[int, int], str] = {}
    for r in range(rows):
        for c in range(cols):
            val = int(game_state[r, c])
            if not (0 <= val <= 8):
                continue
            cov = [(nr, nc) for nr, nc in _neighbours(r, c, rows, cols)
                   if game_state[nr, nc] == COVERED]
            fl = sum(1 for nr, nc in _neighbours(r, c, rows, cols)
                     if game_state[nr, nc] == FLAGGED)
            if not cov:
                continue
            left = val - fl
            if left == 0:
                for cell in cov:
                    sps_reason.setdefault(
                        cell, f"SPS: angka ({r},{c})={val}, flag={fl} -> sisa bom 0 "
                              f"-> semua tetangga tertutup AMAN, P=0")
            elif left == len(cov):
                for cell in cov:
                    sps_reason.setdefault(
                        cell, f"SPS: angka ({r},{c})={val}, flag={fl}, tertutup={len(cov)} "
                              f"-> sisa bom = jumlah tertutup -> semua BOM, P=1")
    for cell in safe_sps:
        info["cells"][f"{cell[0]},{cell[1]}"] = {
            "type": "sps", "p": 0.0, "text": sps_reason.get(cell, "SPS: aman pasti")}
    for cell in mine_sps:
        info["cells"][f"{cell[0]},{cell[1]}"] = {
            "type": "sps", "p": 1.0, "text": sps_reason.get(cell, "SPS: bom pasti")}

    # --- Enumerasi per komponen -------------------------------------------
    exact: List[Tuple[Dict, Dict[int, Tuple[int, List[int]]]]] = []
    approx_cells: Dict[Tuple[int, int], float] = {}
    approx_expected = 0.0
    comps = solver._split_components(bordering, constraints, rows, cols) if bordering else []
    for comp in comps:
        try:
            res = solver._enumerate_component(comp, work, rows, cols)
        except _BudgetExceeded:
            est = solver._local_estimate(comp, work, rows, cols)
            for cell in comp["border"]:
                approx_cells[cell] = est.get(cell, 0.5)
                approx_expected += approx_cells[cell]
            info["approx"] = True
            continue
        if not res:
            for cell in comp["border"]:
                approx_cells[cell] = 0.5
            continue
        exact.append((comp, res))

    R2 = max(0, R - int(round(approx_expected)))

    def tail(m: int, use: bool) -> int:
        if not use:
            return 1
        k = R2 - m
        return comb(N, k) if 0 <= k <= N else 0

    dists = [{k: v[0] for k, v in res.items()} for _, res in exact]
    full = _convolve(dists)
    use_prior = True
    Z = sum(v * tail(m, True) for m, v in full.items())
    if Z == 0:
        use_prior = False
        Z = sum(full.values())
    info["Z"] = _fmt(Z)
    info["use_prior"] = use_prior

    for m in sorted(full):
        info["mix"].append({"m": m, "ways": _fmt(full[m]),
                            "tail": _fmt(tail(m, use_prior)),
                            "weight": _fmt(full[m] * tail(m, use_prior))})

    # --- Rincian per komponen & per sel -----------------------------------
    for ci, (comp, res) in enumerate(exact):
        others = _convolve(dists[:ci] + dists[ci + 1:])
        w = {k: sum(v * tail(k + m, use_prior) for m, v in others.items()) for k in res}
        clues = []
        for (cr, cc) in comp["constraints"]:
            nb = [(nr, nc) for nr, nc in _neighbours(cr, cc, rows, cols)
                  if (nr, nc) in set(comp["border"])]
            fl = sum(1 for nr, nc in _neighbours(cr, cc, rows, cols) if work[nr, nc] == FLAGGED)
            clues.append({"r": cr, "c": cc, "value": int(work[cr, cc]),
                          "need": int(work[cr, cc]) - fl,
                          "cells": [list(x) for x in nb]})
        info["components"].append({
            "id": ci + 1,
            "cells": [list(x) for x in comp["border"]],
            "clues": clues,
            "dist": [{"k": k, "solutions": _fmt(res[k][0]), "weight": _fmt(w[k])}
                     for k in sorted(res)],
            "n_solutions": _fmt(sum(v[0] for v in res.values())),
        })
        for j, cell in enumerate(comp["border"]):
            terms = []
            num = 0
            for k in sorted(res):
                nm = res[k][1][j]
                t = nm * w[k]
                num += t
                terms.append({"k": k, "n_mine": nm, "n_sol": _fmt(res[k][0]),
                              "w": _fmt(w[k]), "term": _fmt(t)})
            p = float(num / Z) if Z else 0.5
            info["cells"][f"{cell[0]},{cell[1]}"] = {
                "type": "border", "comp": ci + 1, "p": p,
                "terms": terms, "num": _fmt(num), "Z": _fmt(Z)}

    for cell, p in approx_cells.items():
        info["cells"][f"{cell[0]},{cell[1]}"] = {
            "type": "approx", "p": float(p),
            "text": "Komponen terlalu besar: estimasi lokal rata-rata (sisa bom/tetangga)."}

    # --- Interior (non-border) -------------------------------------------
    if N > 0:
        if exact and use_prior and Z:
            e_int = sum(v * tail(m, True) * (R2 - m) for m, v in full.items())
            p_bg = float(e_int / Z) / N
            text = (f"Interior: P = E[sisa bom]/N = ({_fmt(e_int)}/{_fmt(Z)})/{N}")
        else:
            bm = sum(c["p"] for k, c in info["cells"].items()
                     if c["type"] in ("border", "approx"))
            p_bg = (R2 - bm) / N
            text = f"Interior: P = (R - bom_perbatasan)/N = ({R2} - {bm:.2f})/{N}"
        p_bg = float(min(max(p_bg, 0.0), 1.0))
        for cell in non_border:
            info["cells"][f"{cell[0]},{cell[1]}"] = {
                "type": "interior", "p": p_bg, "text": text}
    return info
