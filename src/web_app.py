"""
web_app.py
==========

Server web lokal (hanya library standar Python) untuk UI pixel-art Minesweeper.
Frontend ada di web/index.html; semua logika game + AI Bayesian tetap di Python.

Jalankan dari root proyek:
    python src/web_app.py            # lalu buka http://127.0.0.1:8000
    python src/web_app.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import numpy as np

try:
    from .minesweeper_app import MinesweeperGame, COVERED, FLAGGED, MINE, OUT_OF_BOARD
    from .bayesian_solver import BayesianMinesweeperSolver
    from .explain import explain
except ImportError:  # dijalankan langsung: python src/web_app.py
    from minesweeper_app import MinesweeperGame, COVERED, FLAGGED, MINE, OUT_OF_BOARD
    from bayesian_solver import BayesianMinesweeperSolver
    from explain import explain

PRESETS = {
    "beginner":     (9, 9, 10),
    "intermediate": (16, 16, 40),
    "expert":       (16, 30, 99),
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_HTML = os.path.join(ROOT, "web", "index.html")


class Session:
    """Satu game aktif + solver. Klik pertama selalu aman (bom diletakkan setelahnya)."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.solver = BayesianMinesweeperSolver()
        self.new("beginner")

    # ---------------------------------------------------------------- #
    def new(self, difficulty: str = "beginner", seed: Optional[int] = None) -> Dict[str, Any]:
        if difficulty not in PRESETS:
            raise ValueError(f"difficulty tidak dikenal: {difficulty}")
        rows, cols, mines = PRESETS[difficulty]
        self.difficulty = difficulty
        self.game = MinesweeperGame(rows, cols, mines)
        self.rng = np.random.default_rng(seed)
        self.started = False
        self.t0: Optional[float] = None
        self.t_end: Optional[float] = None
        return self.snapshot()

    def _check(self, r: Any, c: Any) -> None:
        if not (isinstance(r, int) and isinstance(c, int)
                and 0 <= r < self.game.num_rows and 0 <= c < self.game.num_cols):
            raise ValueError("koordinat tidak valid")

    def _finish_if_over(self) -> None:
        if self.game.game_status != "PLAYING" and self.t_end is None:
            self.t_end = time.time()

    # ---------------------------------------------------------------- #
    def reveal(self, r: int, c: int) -> Dict[str, Any]:
        self._check(r, c)
        g = self.game
        if g.game_status == "PLAYING":
            if not self.started:
                g.load_game(r, c, rng=self.rng)
                self.started = True
                self.t0 = time.time()
            else:
                g.uncover(r, c)
            self._finish_if_over()
        return self.snapshot()

    def flag(self, r: int, c: int) -> Dict[str, Any]:
        self._check(r, c)
        if self.started:
            self.game.toggle_flag(r, c)
        return self.snapshot()

    def hint(self) -> Dict[str, Any]:
        g = self.game
        hint = None
        if g.game_status == "PLAYING":
            if not self.started:
                hint = {"r": g.num_rows // 2, "c": g.num_cols // 2,
                        "p": None, "reason": "FIRST_MOVE", "probs": None}
            else:
                r, c, p, pm, reason = self.solver.recommend_move(g.game_state, g.num_mines)
                if r >= 0:
                    probs = [[None if np.isnan(x) else round(float(x), 4) for x in row]
                             for row in pm]
                    hint = {"r": int(r), "c": int(c), "p": float(p),
                            "reason": reason, "probs": probs,
                            "explain": explain(self.solver, g.game_state, g.num_mines)}
        snap = self.snapshot()
        snap["hint"] = hint
        return snap

    # ---------------------------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        g = self.game
        over = g.game_status != "PLAYING"
        if self.started and self.t0 is not None:
            elapsed = int((self.t_end or time.time()) - self.t0)
        else:
            elapsed = 0
        snap: Dict[str, Any] = {
            "difficulty": self.difficulty,
            "rows": g.num_rows,
            "cols": g.num_cols,
            "mines": g.num_mines,
            "flags": g.count_flagged(),
            "status": g.game_status,
            "started": self.started,
            "elapsed": min(elapsed, 999),
            "state": g.game_state.tolist(),
            # kode encoding dikirim dari Python agar frontend tidak perlu
            # menulis angka literal (sumber tunggal: bayesian_solver.py)
            "codes": {"covered": COVERED, "flagged": FLAGGED,
                      "mine": MINE, "out": OUT_OF_BOARD},
            "exploded": None,
            "mine_cells": [],
        }
        if over:
            snap["mine_cells"] = [[int(r), int(c)] for r, c in np.argwhere(g.board_values == MINE)]
            boom = np.argwhere(g.game_state == MINE)
            if len(boom):
                snap["exploded"] = [int(boom[0][0]), int(boom[0][1])]
        return snap


SESSION = Session()


class Handler(BaseHTTPRequestHandler):
    server_version = "MinesweeperBayes/1.1"

    def log_message(self, fmt, *args):  # senyap
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0] in ("/", "/index.html"):
            try:
                with open(INDEX_HTML, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, b"web/index.html tidak ditemukan", "text/plain; charset=utf-8")
        elif self.path == "/api/state":
            with SESSION.lock:
                self._send(200, json.dumps(SESSION.snapshot()).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
            with SESSION.lock:
                if self.path == "/api/new":
                    out = SESSION.new(data.get("difficulty", "beginner"), data.get("seed"))
                elif self.path == "/api/reveal":
                    out = SESSION.reveal(data.get("r"), data.get("c"))
                elif self.path == "/api/flag":
                    out = SESSION.flag(data.get("r"), data.get("c"))
                elif self.path == "/api/hint":
                    out = SESSION.hint()
                else:
                    return self._send(404, b'{"error":"not found"}', "application/json")
            self._send(200, json.dumps(out).encode(), "application/json")
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Minesweeper Bayesian - web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Minesweeper berjalan di {url}  (Ctrl+C untuk berhenti)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBerhenti.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()