"""
minesweeper_app.py
==================

Complete Minesweeper game engine + Bayesian AI visualization + interactive UI.

Mengikuti arsitektur arXiv:2212.10446:
    board_values : 2D array, -1 = mine, 0..8 = clue counts
    game_state   : 2D array, -2 = covered, -3 = flagged, 0..8 = revealed clue

Komponen:
    - class MinesweeperGame  : engine permainan (murni Python + numpy)
    - plot_board_and_probabilities(...) : visualisasi 2-panel
    - run_interactive_ui(...) : ipywidgets (Jupyter)
    - run_cli(...)            : fallback CLI (terminal / notebook tanpa widget)
    - main()                  : entry point demo

Cara pakai
----------
1) Sebagai script:
       python src/minesweeper_app.py

2) Di Jupyter Notebook:
       from src.minesweeper_app import (
           MinesweeperGame, plot_board_and_probabilities,
           run_interactive_ui, run_cli,
       )
       from src.bayesian_solver import BayesianMinesweeperSolver

       game = MinesweeperGame(num_rows=9, num_cols=9, num_mines=10)
       game.load_game(start_row=4, start_col=4)

       solver = BayesianMinesweeperSolver()
       r, c, p, pm, reason = solver.recommend_move(game.game_state, game.num_mines)
       plot_board_and_probabilities(game, pm, (r, c))
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

# --- opsional: untuk UI interaktif di Jupyter ---
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False

# --- solver (proyek lokal) ---
try:
    from .bayesian_solver import BayesianMinesweeperSolver  # type: ignore
except ImportError:  # dijalankan langsung: python src/minesweeper_app.py
    from bayesian_solver import BayesianMinesweeperSolver  # type: ignore


# --------------------------------------------------------------------------- #
# Constants (encoding sesuai arXiv:2212.10446)
# --------------------------------------------------------------------------- #
COVERED: int = -2
FLAGGED: int = -3
MINE: int = -1

# warna klasik Minesweeper untuk angka 1..8 (dipakai di plot kiri)
NUMBER_COLORS = {
    1: "#1976D2",  # biru
    2: "#388E3C",  # hijau tua
    3: "#D32F2F",  # merah
    4: "#512DA8",  # ungu tua
    5: "#8D6E63",  # cokelat
    6: "#00838F",  # teal
    7: "#212121",  # hitam
    8: "#757575",  # abu-abu
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
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


def _compute_neighbor_counts(raw: np.ndarray) -> np.ndarray:
    """Hitung jumlah bom tetangga untuk setiap sel; sel bom tetap -1."""
    rows, cols = raw.shape
    counts = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if raw[r, c] == MINE:
                counts[r, c] = MINE
                continue
            cnt = 0
            for nr, nc in _neighbors(r, c, rows, cols):
                if raw[nr, nc] == MINE:
                    cnt += 1
            counts[r, c] = cnt
    return counts


def parse_board_text(
    text: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
    """
    Parser tunggal untuk input board manual (dipakai UI, CLI, dan dataset).

    Simbol (dipisah spasi/koma):
        0..8            = sel terungkap
        ?  _  -2        = tertutup (tidak diketahui)
        F  -3           = flag
        M  *  -1        = bom (ground truth; sel ini tetap tertutup di state)
        '# mines=N'     = total bom (opsional)

    Returns
    -------
    game_state   : array state pemain
    board_values : array ground truth, atau None bila tidak ada simbol bom
    total_mines  : int, atau None bila tak bisa ditentukan
    """
    total_mines: Optional[int] = None
    rows_data: List[List[int]] = []
    for line in text.strip().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "mines=" in s:
                try:
                    total_mines = int(s.split("mines=")[1].split()[0])
                except ValueError:
                    raise ValueError(f"Nilai mines tidak valid: '{s}'")
            continue
        row: List[int] = []
        for tok in s.replace(",", " ").split():
            tu = tok.upper()
            if tu in ("F", "-3"):
                row.append(FLAGGED)
            elif tu in ("?", "_", "-2"):
                row.append(COVERED)
            elif tu in ("M", "*", "-1"):
                row.append(MINE)
            else:
                try:
                    v = int(tok)
                except ValueError:
                    raise ValueError(f"Token tidak dikenal: '{tok}'")
                if not 0 <= v <= 8:
                    raise ValueError(f"Angka clue harus 0..8, dapat {v}")
                row.append(v)
        rows_data.append(row)

    if not rows_data:
        raise ValueError("Board kosong.")
    if len({len(r) for r in rows_data}) != 1:
        raise ValueError("Panjang baris tidak konsisten.")

    arr = np.array(rows_data, dtype=int)
    has_truth = bool(np.any(arr == MINE))
    board_values: Optional[np.ndarray] = None
    game_state = arr.copy()
    if has_truth:
        board_values = _compute_neighbor_counts(np.where(arr == MINE, MINE, 0))
        game_state[arr == MINE] = COVERED
        revealed = arr >= 0
        if np.any(board_values[revealed] != arr[revealed]):
            raise ValueError("Angka clue tidak cocok dengan posisi bom (M).")
        if total_mines is None:
            total_mines = int(np.sum(arr == MINE))
    elif total_mines is None:
        flags = int(np.sum(arr == FLAGGED))
        total_mines = flags if flags else max(1, round(arr.size * 0.15))
    return game_state, board_values, total_mines


# --------------------------------------------------------------------------- #
# Game Engine
# --------------------------------------------------------------------------- #
class MinesweeperGame:
    """Mesin permainan Minesweeper."""

    def __init__(self, num_rows: int, num_cols: int, num_mines: int) -> None:
        if num_rows <= 0 or num_cols <= 0:
            raise ValueError("num_rows dan num_cols harus > 0.")
        if not (0 <= num_mines < num_rows * num_cols):
            raise ValueError("num_mines harus 0 <= mines < total sel.")

        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.num_mines = int(num_mines)

        self.board_values = np.zeros((num_rows, num_cols), dtype=int)
        self.game_state = np.full((num_rows, num_cols), COVERED, dtype=int)
        self.game_status: str = "PLAYING"
        self.moves_history: List[Dict[str, Any]] = []
        # False bila board berasal dari state saja (posisi bom asli tak diketahui)
        self.has_ground_truth: bool = True

        self._total_safe = num_rows * num_cols - num_mines
        self._revealed_safe = 0
        self._move_counter = 0

    def load_game(self, start_row: int, start_col: int,
                  rng: Optional[np.random.Generator] = None) -> None:
        """Letakkan bom secara random, dijamin first-click aman."""
        if not (0 <= start_row < self.num_rows and 0 <= start_col < self.num_cols):
            raise ValueError("start_row/start_col di luar board.")

        if rng is None:
            rng = np.random.default_rng()

        self.board_values = np.zeros((self.num_rows, self.num_cols), dtype=int)
        self.game_state = np.full((self.num_rows, self.num_cols), COVERED, dtype=int)
        self.game_status = "PLAYING"
        self.moves_history.clear()
        self.has_ground_truth = True
        self._revealed_safe = 0
        self._move_counter = 0

        forbidden = {(start_row, start_col)}
        for nr, nc in _neighbors(start_row, start_col, self.num_rows, self.num_cols):
            forbidden.add((nr, nc))

        candidates = [
            (r, c) for r in range(self.num_rows) for c in range(self.num_cols)
            if (r, c) not in forbidden
        ]
        
        if len(candidates) < self.num_mines:
            forbidden = {(start_row, start_col)}
            candidates = [
                (r, c) for r in range(self.num_rows) for c in range(self.num_cols)
                if (r, c) not in forbidden
            ]
        if len(candidates) < self.num_mines:
            raise ValueError("Board terlalu kecil untuk jumlah bom yang diminta.")

        chosen_idx = rng.choice(len(candidates), size=self.num_mines, replace=False)
        for idx in chosen_idx:
            r, c = candidates[int(idx)]
            self.board_values[r, c] = MINE

        self.board_values = _compute_neighbor_counts(self.board_values)
        self.uncover(start_row, start_col)

    def load_custom_board(self, board_values: Optional[np.ndarray],
                          initial_game_state: Optional[np.ndarray] = None,
                          num_mines: Optional[int] = None) -> None:
        """
        Setup dari input manual / dataset.

        board_values = None berarti posisi bom asli tidak diketahui (hanya
        state pemain). Solver tetap bisa merekomendasikan langkah, tetapi
        uncover() tidak bisa dijalankan.
        """
        if board_values is None:
            if initial_game_state is None or num_mines is None:
                raise ValueError(
                    "Tanpa board_values, initial_game_state dan num_mines wajib.")
            gs = np.array(initial_game_state, dtype=int)
            if gs.ndim != 2:
                raise ValueError("initial_game_state harus 2D.")
            self.has_ground_truth = False
            self.num_rows, self.num_cols = gs.shape
            self.num_mines = int(num_mines)
            self.board_values = np.zeros_like(gs)
            self.game_state = gs.copy()
        else:
            arr = np.array(board_values, dtype=int)
            if arr.ndim != 2:
                raise ValueError("board_values harus 2D.")
            if not set(np.unique(arr).tolist()).issubset(set(range(-1, 9))):
                raise ValueError("board_values mengandung nilai tak valid.")
            self.has_ground_truth = True
            self.num_rows, self.num_cols = arr.shape
            self.num_mines = (int(num_mines) if num_mines is not None
                              else int(np.sum(arr == MINE)))
            self.board_values = arr
            if initial_game_state is None:
                self.game_state = np.full(arr.shape, COVERED, dtype=int)
            else:
                gs = np.array(initial_game_state, dtype=int)
                if gs.shape != arr.shape:
                    raise ValueError("initial_game_state shape tidak cocok.")
                self.game_state = gs.copy()

        if not (0 <= self.num_mines <= self.num_rows * self.num_cols):
            raise ValueError("num_mines di luar jangkauan.")
        self._total_safe = self.num_rows * self.num_cols - self.num_mines
        self.game_status = "PLAYING"
        self.moves_history.clear()
        self._revealed_safe = int(np.sum(self.game_state >= 0))
        self._move_counter = 0
        if self.has_ground_truth and self._revealed_safe == self._total_safe:
            self.game_status = "WON"

    def uncover(self, row: int, col: int) -> Dict[str, Any]:
        """Buka sel (row, col)."""
        if not (0 <= row < self.num_rows and 0 <= col < self.num_cols):
            raise ValueError("Koordinat di luar board.")
        if not self.has_ground_truth:
            raise RuntimeError(
                "Posisi bom asli tidak diketahui (board dari state saja); "
                "uncover() tidak bisa dijalankan.")
        self._move_counter += 1
        info: Dict[str, Any] = {
            "turn": self._move_counter, "action": "uncover",
            "row": int(row), "col": int(col), "result": "noop", "revealed_cells": [],
        }

        if self.game_status != "PLAYING":
            info["result"] = "game_over"
            self.moves_history.append(info)
            return info

        cur = self.game_state[row, col]
        if cur == FLAGGED:
            info["result"] = "flagged"
            self.moves_history.append(info)
            return info
        if cur != COVERED:
            info["result"] = "already_revealed"
            self.moves_history.append(info)
            return info

        if self.board_values[row, col] == MINE:
            self.game_state[row, col] = MINE
            self.game_status = "LOST"
            info["result"] = "mine"
            self.moves_history.append(info)
            return info

        queue: deque = deque([(row, col)])
        revealed: List[Tuple[int, int]] = []
        
        while queue:
            r, c = queue.popleft()
            if self.game_state[r, c] != COVERED:
                continue
            if self.board_values[r, c] == MINE:
                continue
            
            self.game_state[r, c] = int(self.board_values[r, c])
            self._revealed_safe += 1
            revealed.append((r, c))
            
            if self.board_values[r, c] == 0:
                for nr, nc in _neighbors(r, c, self.num_rows, self.num_cols):
                    if self.game_state[nr, nc] == COVERED:
                        queue.append((nr, nc))

        info["revealed_cells"] = revealed
        info["result"] = "safe" if revealed else "noop"

        if self._revealed_safe == self._total_safe:
            self.game_status = "WON"

        self.moves_history.append(info)
        return info

    def toggle_flag(self, row: int, col: int) -> bool:
        """Toggle flag (hanya pada sel COVERED)."""
        if self.game_status != "PLAYING":
            return False
        if not (0 <= row < self.num_rows and 0 <= col < self.num_cols):
            return False
            
        cur = self.game_state[row, col]
        if cur == COVERED:
            self.game_state[row, col] = FLAGGED
            self.moves_history.append({
                "turn": self._move_counter + 1, "action": "flag",
                "row": int(row), "col": int(col), "result": "flagged", "revealed_cells": [],
            })
            self._move_counter += 1
            return True
        if cur == FLAGGED:
            self.game_state[row, col] = COVERED
            self.moves_history.append({
                "turn": self._move_counter + 1, "action": "unflag",
                "row": int(row), "col": int(col), "result": "unflagged", "revealed_cells": [],
            })
            self._move_counter += 1
            return True
        return False

    def count_covered(self) -> int:
        return int(np.sum(self.game_state == COVERED))

    def count_flagged(self) -> int:
        return int(np.sum(self.game_state == FLAGGED))

    def progress(self) -> float:
        if self._total_safe <= 0:
            return 1.0
        return float(self._revealed_safe / self._total_safe)

    def __repr__(self) -> str:
        return (
            f"MinesweeperGame(rows={self.num_rows}, cols={self.num_cols}, "
            f"mines={self.num_mines}, status={self.game_status}, "
            f"progress={self.progress() * 100:.1f}%)"
        )

    def __str__(self) -> str:
        """CLI-friendly rendering."""
        lines = []
        header = "    " + "".join(f"{c:>4}" for c in range(self.num_cols))
        lines.append(header)
        for r in range(self.num_rows):
            row_tokens = []
            for c in range(self.num_cols):
                v = self.game_state[r, c]
                if v == COVERED:
                    row_tokens.append("  ? ")
                elif v == FLAGGED:
                    row_tokens.append("  F ")
                elif v == MINE:
                    row_tokens.append("  * ")
                elif v == 0:
                    row_tokens.append("  . ")
                else:
                    row_tokens.append(f"  {int(v)} ")
            lines.append(f"{r:>3} " + "".join(row_tokens))
        lines.append(
            f"\nStatus: {self.game_status}  "
            f"| Mines: {self.num_mines}  "
            f"| Flags: {self.count_flagged()}  "
            f"| Progress: {self.progress() * 100:.1f}%"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def plot_board_and_probabilities(
    game: MinesweeperGame,
    prob_matrix: Optional[np.ndarray] = None,
    recommended_move: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (14, 6),
    show: bool = True,
):
    """Plot dua panel: board permainan & heatmap probabilitas Bayes."""
    if not HAS_MPL:
        raise ImportError("matplotlib tidak terpasang.")

    rows, cols = game.num_rows, game.num_cols
    fig, (ax_board, ax_prob) = plt.subplots(1, 2, figsize=figsize)

    # ---- Panel kiri: board ---- #
    ax_board.set_facecolor("#BDBDBD")
    for r in range(rows):
        for c in range(cols):
            v = int(game.game_state[r, c])
            if v == COVERED:
                face, txt, txt_color = "#9E9E9E", "", "white"
            elif v == FLAGGED:
                face, txt, txt_color = "#9E9E9E", "⚑", "#D32F2F"
            elif v == MINE:
                face, txt, txt_color = "#E57373", "✸", "black"
            elif v == 0:
                face, txt, txt_color = "#E0E0E0", "", "black"
            else:
                face, txt, txt_color = "#E0E0E0", str(v), NUMBER_COLORS.get(v, "black")
                
            ax_board.add_patch(Rectangle((c, r), 1, 1, facecolor=face, edgecolor="#616161", linewidth=0.8))
            if txt:
                ax_board.text(c + 0.5, r + 0.5, txt, ha="center", va="center",
                              fontsize=12, fontweight="bold", color=txt_color)

    if recommended_move is not None:
        rr, rc = recommended_move
        if 0 <= rr < rows and 0 <= rc < cols:
            ax_board.add_patch(Rectangle((rc, rr), 1, 1, fill=False, edgecolor="#00E676", linewidth=3.5, zorder=5))

    ax_board.set_xlim(0, cols)
    ax_board.set_ylim(rows, 0)
    ax_board.set_xticks(np.arange(0.5, cols + 0.5, 1))
    ax_board.set_yticks(np.arange(0.5, rows + 0.5, 1))
    ax_board.set_xticklabels([])
    ax_board.set_yticklabels([])
    ax_board.set_aspect("equal")
    ax_board.set_title(f"Board — {game.game_status}", fontsize=12)

    # ---- Panel kanan: heatmap ---- #
    if prob_matrix is None:
        prob_matrix = np.full((rows, cols), np.nan, dtype=float)
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    clipped = np.clip(prob_matrix, 0.0, 1.0)
    masked = np.ma.masked_invalid(clipped)

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#9E9E9E")
    im = ax_prob.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", origin="upper")

    ax_prob.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax_prob.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax_prob.grid(which="minor", color="#424242", linestyle="-", linewidth=0.6)
    ax_prob.tick_params(which="both", length=0, labelbottom=False, labelleft=False)

    for r in range(rows):
        for c in range(cols):
            p = clipped[r, c]
            if np.isnan(p) or game.game_state[r, c] != COVERED:
                continue
            txt = f"{p * 100:.0f}%"
            color = "white" if (p > 0.6 or p < 0.15) else "black"
            ax_prob.text(c, r, txt, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    if recommended_move is not None:
        rr, rc = recommended_move
        if 0 <= rr < rows and 0 <= rc < cols:
            ax_prob.add_patch(Rectangle((rc - 0.5, rr - 0.5), 1, 1, fill=False, edgecolor="#00E676", linewidth=3.0, zorder=5))

    ax_prob.set_title("Bayesian P(Mine | Evidence)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)
    cbar.set_label("P(mine)", rotation=270, labelpad=12)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Interactive runner (ipywidgets)
# --------------------------------------------------------------------------- #
def run_interactive_ui(
    game: MinesweeperGame,
    solver: Optional[BayesianMinesweeperSolver] = None,
    auto_solve_delay: float = 0.0,
) -> Any:
    """Tampilkan UI interaktif berbasis ipywidgets."""
    if not HAS_IPYWIDGETS:
        print("[INFO] ipywidgets tidak tersedia. Gunakan run_cli() sebagai fallback.")
        return None

    if solver is None:
        solver = BayesianMinesweeperSolver()

    out = widgets.Output()
    status = widgets.HTML()

    btn_step = widgets.Button(description="Step AI Move", button_style="primary")
    btn_auto = widgets.Button(description="Auto Solve Game", button_style="success")
    btn_reset = widgets.Button(description="Reset")
    btn_manual = widgets.Button(description="Manual Input", button_style="warning")

    manual_box = widgets.Textarea(
        value="",
        placeholder="Tempel board manual:\n? = covered, F = flag, 0..8 = clue, M = bom (opsional)\nBaris '# mines=N' menentukan total bom.",
        description="Board:",
        layout=widgets.Layout(width="100%", height="140px"),
        disabled=True,
    )
    btn_apply = widgets.Button(description="Apply Manual Board", button_style="danger", disabled=True)

    state: Dict[str, Any] = {"last_prob": None, "last_move": None}

    def _update_status(msg: str = "") -> None:
        status.value = (
            f"<b>Status:</b> {game.game_status} &nbsp;|&nbsp; "
            f"<b>Mines:</b> {game.num_mines} &nbsp;|&nbsp; "
            f"<b>Flags:</b> {game.count_flagged()} &nbsp;|&nbsp; "
            f"<b>Progress:</b> {game.progress() * 100:.1f}%<br>"
            f"<span style='color:#555'>{msg}</span>"
        )

    def _render(msg: str = "") -> None:
        with out:
            clear_output(wait=True)
            plot_board_and_probabilities(game, state.get("last_prob"), state.get("last_move"), show=True)
        _update_status(msg)

    def _ai_step() -> str:
        if game.game_status != "PLAYING":
            return f"Game sudah berakhir: {game.game_status}."
        r, c, p, pm, reason = solver.recommend_move(game.game_state, game.num_mines)
        if r < 0 or c < 0:
            return "Tidak ada kandidat langkah."
        state["last_prob"] = pm
        state["last_move"] = (r, c)
        if not game.has_ground_truth:
            return (f"Rekomendasi AI: ({r},{c}) p={p * 100:.1f}% [{reason}] "
                    "(board manual tanpa bom asli: langkah tidak dijalankan)")
        info = game.uncover(r, c)
        return f"AI pilih ({r},{c}) p={p * 100:.1f}% [{reason}] → {info['result']}"

    def on_step(_btn) -> None:
        _render(_ai_step())

    def on_auto(_btn) -> None:
        if not game.has_ground_truth:
            _render("Auto-solve butuh posisi bom asli (tulis bom dengan 'M').")
            return
        safety = 0
        while game.game_status == "PLAYING" and safety < game.num_rows * game.num_cols * 3:
            _ai_step()
            safety += 1
            if auto_solve_delay > 0:
                time.sleep(auto_solve_delay)
        _render(f"Auto-solve selesai: {game.game_status}")

    def on_reset(_btn) -> None:
        cr, cc = game.num_rows // 2, game.num_cols // 2
        game.load_game(cr, cc)
        state["last_prob"] = None
        state["last_move"] = None
        manual_box.disabled = True
        btn_apply.disabled = True
        _render("Game baru dimuat.")

    def on_manual_toggle(_btn) -> None:
        manual_box.disabled = not manual_box.disabled
        btn_apply.disabled = manual_box.disabled
        _update_status("Manual input aktif." if not manual_box.disabled else "Manual input dimatikan.")

    def on_apply(_btn) -> None:
        try:
            gs, bv, tm = parse_board_text(manual_box.value)
            game.load_custom_board(bv, gs, num_mines=tm)
            state["last_prob"] = None
            state["last_move"] = None
            _render("Manual board dimuat.")
        except Exception as e:
            _update_status(f"<span style='color:#B71C1C'>Error: {e}</span>")

    btn_step.on_click(on_step)
    btn_auto.on_click(on_auto)
    btn_reset.on_click(on_reset)
    btn_manual.on_click(on_manual_toggle)
    btn_apply.on_click(on_apply)

    controls = widgets.HBox([btn_step, btn_auto, btn_reset, btn_manual])
    container = widgets.VBox([status, controls, manual_box, btn_apply, out])

    _render("Siap.")
    return container


# --------------------------------------------------------------------------- #
# CLI fallback
# --------------------------------------------------------------------------- #
def run_cli(
    game: MinesweeperGame,
    solver: Optional[BayesianMinesweeperSolver] = None,
    max_steps: int = 200,
) -> None:
    """Fallback CLI (cocok untuk terminal atau notebook tanpa widget)."""
    if solver is None:
        solver = BayesianMinesweeperSolver()

    print("=" * 70)
    print(f"MINESWEEPER CLI — board {game.num_rows}x{game.num_cols}, {game.num_mines} mines")
    print("=" * 70)

    step = 0
    while game.game_status == "PLAYING" and step < max_steps:
        print(game)
        r, c, p, _pm, reason = solver.recommend_move(game.game_state, game.num_mines)
        if r < 0 or c < 0:
            print("[!] Tidak ada kandidat langkah. Berhenti.")
            break
        print(f"\n>> AI memilih ({r}, {c}) P(mine)={p * 100:.1f}% [{reason}]")
        info = game.uncover(r, c)
        print(f"   Hasil: {info['result']} (sel dibuka: {len(info['revealed_cells'])})")
        step += 1
        print()

    print("=" * 70)
    print(game)
    print(f"\nGAME OVER — status: {game.game_status} | langkah: {step}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Demo / entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Demo: mainkan satu game Beginner secara otomatis, lalu plot heatmap."""
    print("[DEMO] Minesweeper Bayesian Inference — script mode")
    print("-" * 60)

    game = MinesweeperGame(num_rows=9, num_cols=9, num_mines=10)
    game.load_game(start_row=4, start_col=4)
    print(f"Board initialized: {game}")

    solver = BayesianMinesweeperSolver()

    r, c, p, pm, reason = solver.recommend_move(game.game_state, game.num_mines)
    print(f"\n[Step 1] AI rekomendasi: ({r},{c}) P(mine)={p * 100:.1f}% [{reason}]")

    if HAS_MPL:
        plot_board_and_probabilities(game, pm, (r, c), show=True)
    else:
        print("[WARN] matplotlib tidak terpasang — skip visualisasi.")

    print("\n[Auto-solve] Menjalankan solver sampai selesai...")
    steps = 0
    while game.game_status == "PLAYING" and steps < 200:
        r, c, _p, _pm, _reason = solver.recommend_move(game.game_state, game.num_mines)
        if r < 0 or c < 0:
            break
        game.uncover(r, c)
        steps += 1

    print(f"Status akhir: {game.game_status} | total langkah AI: {steps}")


if __name__ == "__main__":
    main()