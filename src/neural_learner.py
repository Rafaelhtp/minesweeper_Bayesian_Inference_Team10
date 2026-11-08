"""
neural_learner.py
=================

Pelajar jaringan saraf (MLP) untuk Minesweeper mengikuti jurnal
"Neural Network Learner for Minesweeper" (arXiv:2212.10446), Bab 5.3 - 5.5,
dilatih dari dataset Kaggle `minesweeper_dataset`.

Kesesuaian jurnal <-> dataset <-> kode
--------------------------------------
Jurnal (Bab 5.5.1)           Dataset                      Kode ini
---------------------------  ---------------------------  --------------------------
jendela geser 5x5 -> 24      24 kolom cell_R_C            window_cells() / 24 input
input (sel tengah = target)  (sel tengah tidak ada)
keluaran 0..1 = peluang bom  label `safe` (1 = aman)      target dilatih = 1 - safe,
(Bab 5.5.2, regresi)                                      keluaran = P(bom)
ReLU tersembunyi, sigmoid    -                            NumpyMLP (ReLU + sigmoid)
keluaran, MSE atau XENT
MLP learner3: 70-80-60-35    -                            DEFAULT_HIDDEN
pilih sel terkecil P(bom)    -                            NeuralWindowSolver
pada sel perbatasan
data latih dibangkitkan saat -                            generate_selfplay_windows()
game dimainkan (Bab 5.5.4)
learner tidak boleh flag;    dataset tanpa flag           flag dipetakan ke tertutup,
selalu mengeluarkan sel                                   hanya sel tertutup dipilih
tertutup
encoding sel                 -1 tertutup, -2 luar papan   sama dengan dataset
                                                           (lihat bayesian_solver.py)

Hanya numpy yang dibutuhkan (tanpa TensorFlow / PyTorch / Deeplearning4j).

Pemakaian
---------
    from src.neural_learner import train_from_dataset, NeuralWindowSolver
    model, report = train_from_dataset("minesweeper_dataset_terbaru.xlsx", nrows=300_000)
    model.save("models/mlp_window.npz")

    solver = NeuralWindowSolver.load("models/mlp_window.npz")
    r, c, p, prob_matrix, reason = solver.recommend_move(game_state, total_mines)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from .bayesian_solver import (BayesianMinesweeperSolver, COVERED, FLAGGED,
                                  OUT_OF_BOARD)
except ImportError:  # dijalankan dengan sys.path ke src/
    from bayesian_solver import (BayesianMinesweeperSolver, COVERED, FLAGGED,
                                 OUT_OF_BOARD)

__all__ = ["NumpyMLP", "WindowMLP", "NeuralWindowSolver", "window_cells",
           "encode_inputs", "train_from_dataset", "generate_selfplay_windows",
           "train_from_selfplay", "evaluate_binary", "DEFAULT_HIDDEN"]

# Konfigurasi "MLP learner3" jurnal (Tabel 5.1): 4 lapisan tersembunyi.
DEFAULT_HIDDEN: Tuple[int, ...] = (70, 80, 60, 35)

# Nilai sel yang mungkin: -2 (luar papan), -1 (tertutup), 0..8  -> 11 kategori.
N_CATEGORIES = 11
_CATEGORY_OFFSET = 2          # indeks one-hot = nilai + 2

# Posisi sel tengah pada jendela 5x5 yang diratakan (baris 2, kolom 2).
_CENTER = 12


# --------------------------------------------------------------------------- #
# Jendela 5x5 dari papan permainan
# --------------------------------------------------------------------------- #
def window_cells(game_state: np.ndarray,
                 coords: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Ambil 24 sel di sekitar tiap koordinat (jendela 5x5 tanpa sel tengah).

    Urutan kolom sama dengan KAGGLE_CELL_COLUMNS (cell_-2_-2 ... cell_2_2),
    sel di luar papan bernilai OUT_OF_BOARD (-2) seperti di dataset. Flag
    (-3) dipetakan ke tertutup (-1) karena dataset tidak punya flag dan
    pelajar tidak mengandalkan flag (jurnal Bab 3.2).

    Returns : int8 array (len(coords), 24)  (nilai -2..8, hemat memori)
    """
    rows, cols = game_state.shape
    padded = np.full((rows + 4, cols + 4), OUT_OF_BOARD, dtype=int)
    inner = game_state.astype(int).copy()
    inner[inner == FLAGGED] = COVERED
    inner[inner < OUT_OF_BOARD] = COVERED     # kode lain (mis. bom meledak)
    padded[2:-2, 2:-2] = inner

    out = np.empty((len(coords), 24), dtype=np.int8)
    for i, (r, c) in enumerate(coords):
        win = padded[r:r + 5, c:c + 5].reshape(-1)
        out[i] = np.delete(win, _CENTER)
    return out


def encode_inputs(cells: np.ndarray,
                  density: Optional[np.ndarray] = None,
                  onehot: bool = False,
                  use_density: bool = False) -> np.ndarray:
    """
    Ubah jendela (n, 24) menjadi matriks fitur float32.

    onehot=False : 24 input mentah dibagi 8 (hanya penskalaan), sama seperti
                   MLP jurnal Bab 5.5.1 (24 input).
    onehot=True  : 24 x 11 input biner (kategori -2, -1, 0..8). Jurnal Bab
                   5.6.1.2 menyarankan one-hot untuk CNN karena angka sel
                   adalah label kategori, bukan besaran numerik.
    use_density  : tambahkan `global_density` sebagai fitur terakhir.
    """
    cells = np.asarray(cells, dtype=int)
    if onehot:
        idx = np.clip(cells + _CATEGORY_OFFSET, 0, N_CATEGORIES - 1)
        feats = np.zeros((len(cells), 24, N_CATEGORIES), dtype=np.float32)
        np.put_along_axis(feats, idx[..., None], 1.0, axis=2)
        feats = feats.reshape(len(cells), -1)
    else:
        feats = (cells / 8.0).astype(np.float32)
    if use_density:
        if density is None:
            raise ValueError("use_density=True tetapi density tidak diberikan.")
        feats = np.concatenate(
            [feats, np.asarray(density, dtype=np.float32)[:, None]], axis=1)
    return feats


# --------------------------------------------------------------------------- #
# MLP numpy
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class NumpyMLP:
    """
    MLP fully-connected: ReLU di lapisan tersembunyi, sigmoid di keluaran
    (satu neuron, nilai 0..1). Loss: 'mse' atau 'xent' (cross-entropy),
    optimizer: 'adam' atau 'sgd'. Parameter mengikuti Bab 5.5.3 jurnal.
    """

    def __init__(self, n_inputs: int, hidden: Sequence[int] = DEFAULT_HIDDEN,
                 seed: int = 0) -> None:
        self.sizes = [int(n_inputs)] + [int(h) for h in hidden] + [1]
        rng = np.random.default_rng(seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(self.sizes) - 1):
            fan_in, fan_out = self.sizes[i], self.sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in) if i < len(self.sizes) - 2 \
                else np.sqrt(1.0 / fan_in)
            self.W.append((rng.standard_normal((fan_in, fan_out)) * scale)
                          .astype(np.float32))
            self.b.append(np.zeros(fan_out, dtype=np.float32))

    @property
    def n_parameters(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    # ---- forward / backward ---------------------------------------------- #
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        acts = [X]
        a = X
        for i in range(len(self.W) - 1):
            a = np.maximum(a @ self.W[i] + self.b[i], 0.0)
            acts.append(a)
        out = _sigmoid(a @ self.W[-1] + self.b[-1])
        return acts, out

    def predict(self, X: np.ndarray, batch: int = 65536,
                transform: Optional[Any] = None) -> np.ndarray:
        """
        Keluaran sigmoid (n,) = P(kelas 1). Bila `transform` diberikan, X
        di-encode per potongan (hemat memori untuk one-hot).
        """
        if transform is None:
            X = np.asarray(X, dtype=np.float32)
        res = [self._forward(np.asarray(transform(X[i:i + batch]) if transform
                                        else X[i:i + batch], dtype=np.float32))[1][:, 0]
               for i in range(0, len(X), batch)]
        return np.concatenate(res) if res else np.empty(0, dtype=np.float32)

    def _grads(self, X: np.ndarray, y: np.ndarray, loss: str):
        acts, out = self._forward(X)
        y = y.reshape(-1, 1)
        if loss == "xent":
            delta = (out - y)                       # d loss / d z untuk BCE
        else:                                       # mse
            delta = 2.0 * (out - y) * out * (1.0 - out)
        delta = delta / len(X)
        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        return gW, gb, out

    # ---- pelatihan -------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10,
            batch_size: int = 256, lr: float = 0.001, loss: str = "xent",
            optimizer: str = "adam", seed: int = 0,
            val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True,
            transform: Optional[Any] = None) -> List[Dict[str, float]]:
        if loss not in ("mse", "xent"):
            raise ValueError("loss harus 'mse' atau 'xent'.")
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer harus 'adam' atau 'sgd'.")
        # transform (opsional): fungsi yang meng-encode potongan X per batch,
        # sehingga fitur besar (one-hot) tidak perlu disimpan utuh di memori.
        if transform is None:
            X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        rng = np.random.default_rng(seed)
        n = len(X)
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        history: List[Dict[str, float]] = []
        for ep in range(1, epochs + 1):
            t0 = time.time()
            order = rng.permutation(n)
            total = 0.0
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                xb = X[idx]
                if transform is not None:
                    xb = np.asarray(transform(xb), dtype=np.float32)
                gW, gb, out = self._grads(xb, y[idx], loss)
                t += 1
                for i in range(len(self.W)):
                    if optimizer == "sgd":
                        self.W[i] -= lr * gW[i]
                        self.b[i] -= lr * gb[i]
                    else:
                        mW[i] = beta1 * mW[i] + (1 - beta1) * gW[i]
                        vW[i] = beta2 * vW[i] + (1 - beta2) * gW[i] ** 2
                        mb[i] = beta1 * mb[i] + (1 - beta1) * gb[i]
                        vb[i] = beta2 * vb[i] + (1 - beta2) * gb[i] ** 2
                        c1, c2 = 1 - beta1 ** t, 1 - beta2 ** t
                        self.W[i] -= lr * (mW[i] / c1) / (np.sqrt(vW[i] / c2) + eps)
                        self.b[i] -= lr * (mb[i] / c1) / (np.sqrt(vb[i] / c2) + eps)
                p = np.clip(out[:, 0], 1e-7, 1 - 1e-7)
                yy = y[idx]
                total += float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum())
            rec: Dict[str, float] = {"epoch": ep, "train_logloss": total / n,
                                     "seconds": time.time() - t0}
            if val is not None:
                m = evaluate_binary(val[1], self.predict(val[0], transform=transform))
                rec.update({f"val_{k}": v for k, v in m.items()})
            history.append(rec)
            if verbose:
                extra = (f" | val acc={rec['val_accuracy']:.4f} "
                         f"auc={rec['val_auc']:.4f}") if val is not None else ""
                print(f"  epoch {ep:>2}/{epochs}  train_logloss="
                      f"{rec['train_logloss']:.4f}{extra}  ({rec['seconds']:.1f}s)")
        return history

    # ---- simpan / muat ---------------------------------------------------- #
    def state(self) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {"sizes": np.array(self.sizes)}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"], d[f"b{i}"] = w, b
        return d

    @classmethod
    def from_state(cls, d: Dict[str, np.ndarray]) -> "NumpyMLP":
        sizes = [int(x) for x in d["sizes"]]
        net = cls(sizes[0], sizes[1:-1])
        net.W = [d[f"W{i}"].astype(np.float32) for i in range(len(sizes) - 1)]
        net.b = [d[f"b{i}"].astype(np.float32) for i in range(len(sizes) - 1)]
        return net


# --------------------------------------------------------------------------- #
# Metrik
# --------------------------------------------------------------------------- #
def evaluate_binary(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Akurasi (ambang 0,5), log-loss, dan AUC (rank / Mann-Whitney)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    logloss = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    acc = float(((p >= 0.5) == (y >= 0.5)).mean())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        auc = float("nan")
    else:
        ranks = np.empty(len(p))
        order = np.argsort(p, kind="mergesort")
        sp = p[order]
        # peringkat rata-rata untuk nilai kembar
        uniq, first, counts = np.unique(sp, return_index=True, return_counts=True)
        avg = first + (counts + 1) / 2.0
        ranks[order] = np.repeat(avg, counts)
        auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return {"accuracy": acc, "logloss": logloss, "auc": auc}


# --------------------------------------------------------------------------- #
# Pembungkus: model + konfigurasi encoding
# --------------------------------------------------------------------------- #
class WindowMLP:
    """MLP + konfigurasi fitur (onehot / density) agar bisa disimpan utuh."""

    def __init__(self, hidden: Sequence[int] = DEFAULT_HIDDEN,
                 onehot: bool = False, use_density: bool = False,
                 seed: int = 0) -> None:
        self.hidden = tuple(int(h) for h in hidden)
        self.onehot = bool(onehot)
        self.use_density = bool(use_density)
        n_in = 24 * (N_CATEGORIES if onehot else 1) + (1 if use_density else 0)
        self.net = NumpyMLP(n_in, self.hidden, seed=seed)

    def features(self, cells: np.ndarray,
                 density: Optional[np.ndarray] = None) -> np.ndarray:
        return encode_inputs(cells, density, self.onehot, self.use_density)

    @staticmethod
    def pack(cells: np.ndarray, density: Optional[np.ndarray]) -> np.ndarray:
        """Gabungkan jendela (n,24) dan density (n,) jadi satu array (n,25)."""
        dens = np.zeros(len(cells)) if density is None else np.asarray(density)
        return np.concatenate([np.asarray(cells, dtype=np.float32),
                               dens[:, None].astype(np.float32)], axis=1)

    def transform(self, packed: np.ndarray) -> np.ndarray:
        """Encode array hasil pack() -> fitur MLP (dipakai per batch)."""
        return encode_inputs(packed[:, :24].astype(int), packed[:, 24],
                             self.onehot, self.use_density)

    def predict_mine(self, cells: np.ndarray,
                     density: Optional[np.ndarray] = None) -> np.ndarray:
        """P(sel target adalah bom) untuk tiap jendela."""
        return self.net.predict(self.features(cells, density))

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, onehot=np.array(self.onehot),
                            use_density=np.array(self.use_density),
                            **self.net.state())

    @classmethod
    def load(cls, path: str) -> "WindowMLP":
        d = dict(np.load(path))
        net = NumpyMLP.from_state(d)
        obj = cls(hidden=net.sizes[1:-1], onehot=bool(d["onehot"]),
                  use_density=bool(d["use_density"]))
        obj.net = net
        return obj


# --------------------------------------------------------------------------- #
# Pelatihan dari dataset Kaggle
# --------------------------------------------------------------------------- #
def train_from_dataset(source: Union[str, Any, Tuple[np.ndarray, ...]], nrows: Optional[int] = None,
                       hidden: Sequence[int] = DEFAULT_HIDDEN,
                       onehot: bool = False, use_density: bool = False,
                       epochs: int = 8, batch_size: int = 256,
                       lr: float = 0.001, loss: str = "xent",
                       optimizer: str = "adam", val_frac: float = 0.1,
                       drop_duplicates: bool = True, seed: int = 0,
                       verbose: bool = True) -> Tuple[WindowMLP, Dict[str, Any]]:
    """
    Latih WindowMLP dari dataset Kaggle (CSV / XLSX / DataFrame, atau tuple
    array (cells, density, safe) dari DatasetLoader.kaggle_arrays).

    Label dilatih sebagai P(bom) = 1 - safe, sesuai keluaran jurnal Bab 5.5.2.
    PERINGATAN: baris dataset TIDAK diacak. Pada ~300 ribu baris pertama hanya
    ~62,7% sampel yang aman, sedangkan baris sisanya ~85% (total 80,4%). Karena
    itu `nrows` (mengambil N baris pertama) memberi sampel yang bias; biarkan
    nrows=None (semua baris) atau beri array yang sudah diacak.

    Baris duplikat dibuang SEBELUM membagi data latih/validasi supaya baris
    yang sama tidak muncul di kedua sisi (kebocoran data). Dataset mengandung
    sekitar 21,5% baris duplikat.

    Returns
    -------
    model  : WindowMLP terlatih
    report : dict berisi jumlah data, metrik validasi, baseline mayoritas, riwayat
    """
    try:
        from .data_and_testing import DatasetLoader
    except ImportError:
        from data_and_testing import DatasetLoader

    if isinstance(source, tuple):          # (cells, density, safe) sudah berupa array
        cells, density, safe = (np.asarray(a) for a in source)
        if nrows is not None:
            cells, density, safe = cells[:nrows], density[:nrows], safe[:nrows]
    else:
        cells, density, safe = DatasetLoader.kaggle_arrays(source, nrows)
    if (safe < 0).any():
        raise ValueError("Dataset tidak punya kolom label 'safe'.")
    n_raw = len(cells)

    if drop_duplicates:
        key = np.concatenate([cells, np.round(density * 1e6).astype(int)[:, None],
                              safe[:, None]], axis=1)
        _, keep = np.unique(key, axis=0, return_index=True)
        keep.sort()
        cells, density, safe = cells[keep], density[keep], safe[keep]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cells))
    n_val = int(len(cells) * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    mine = (1 - safe).astype(np.float32)               # 1 = bom
    model = WindowMLP(hidden, onehot, use_density, seed=seed)
    Xtr = model.pack(cells[tr_idx], density[tr_idx])
    Xva = model.pack(cells[val_idx], density[val_idx])

    if verbose:
        print(f"[train] baris dibaca={n_raw:,} | setelah dedup={len(cells):,} | "
              f"latih={len(tr_idx):,} | validasi={len(val_idx):,} | "
              f"parameter={model.net.n_parameters:,}")
    history = model.net.fit(Xtr, mine[tr_idx], epochs=epochs,
                            batch_size=batch_size, lr=lr, loss=loss,
                            optimizer=optimizer, seed=seed,
                            val=(Xva, mine[val_idx]), verbose=verbose,
                            transform=model.transform)

    p_val = model.net.predict(Xva, transform=model.transform)
    metrics = evaluate_binary(mine[val_idx], p_val)
    majority_acc = float(max(mine[val_idx].mean(), 1 - mine[val_idx].mean()))
    report = {
        "rows_read": int(n_raw), "rows_after_dedup": int(len(cells)),
        "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)),
        "n_parameters": model.net.n_parameters,
        "val_metrics": metrics, "majority_baseline_accuracy": majority_acc,
        "mine_rate_val": float(mine[val_idx].mean()), "history": history,
        "config": {"hidden": list(hidden), "onehot": onehot,
                   "use_density": use_density, "epochs": epochs, "lr": lr,
                   "loss": loss, "optimizer": optimizer, "batch_size": batch_size},
    }
    return model, report


# --------------------------------------------------------------------------- #
# Data latih ala jurnal: dibangkitkan saat game dimainkan (Bab 5.5.4)
# --------------------------------------------------------------------------- #
def generate_selfplay_windows(n_games: int = 3000, rows: int = 9, cols: int = 9,
                              mines: int = 10, seed: int = 0,
                              player: Optional[Any] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mainkan `n_games` game dan rekam jendela 5x5 dari SETIAP sel perbatasan di
    setiap langkah, dengan label = posisi bom sebenarnya (ground truth).

    Ini cara jurnal membangun data latih (Bab 5.5.4): tidak ada dataset
    tetap, sampel dibangkitkan saat game dimainkan. Pemain default adalah
    BayesianMinesweeperSolver sehingga trajektorinya realistis dan game
    berlangsung sampai selesai.

    Mengapa perlu selain dataset Kaggle: pada dataset hanya ~19,6% target
    yang bom, sedangkan di antara sel perbatasan pada game nyata ~51% adalah
    bom. Sel yang harus dinilai saat bermain berasal dari distribusi game,
    bukan distribusi dataset.

    Returns : cells (n, 24), density (n,), mine (n,) dengan 1 = bom
    """
    try:
        from .minesweeper_app import MinesweeperGame, MINE
    except ImportError:
        from minesweeper_app import MinesweeperGame, MINE

    player = player or BayesianMinesweeperSolver()
    helper = BayesianMinesweeperSolver()
    X: List[np.ndarray] = []
    Y: List[float] = []
    for gi in range(n_games):
        rng = np.random.default_rng(seed + gi)
        game = MinesweeperGame(rows, cols, mines)
        game.load_game(int(rng.integers(rows)), int(rng.integers(cols)), rng=rng)
        while game.game_status == "PLAYING":
            border, _cons, _nb = helper.get_bordering_and_constraints(game.game_state)
            border = sorted(cell for cell in border if game.game_state[cell] == COVERED)
            if border:
                X.append(window_cells(game.game_state, border))
                Y.extend(1.0 if game.board_values[r, c] == MINE else 0.0
                         for r, c in border)
            r, c, *_ = player.recommend_move(game.game_state, mines)
            if r < 0:
                break
            game.uncover(r, c)
    cells = np.concatenate(X) if X else np.empty((0, 24), dtype=int)
    density = np.full(len(cells), mines / float(rows * cols))
    return cells, density, np.array(Y, dtype=np.float32)


def train_from_selfplay(n_games: int = 3000, rows: int = 9, cols: int = 9,
                        mines: int = 10, hidden: Sequence[int] = DEFAULT_HIDDEN,
                        onehot: bool = True, use_density: bool = False,
                        epochs: int = 8, batch_size: int = 256,
                        lr: float = 0.001, loss: str = "xent",
                        optimizer: str = "adam", n_test_games: int = 300,
                        seed: int = 0, verbose: bool = True
                        ) -> Tuple[WindowMLP, Dict[str, Any]]:
    """
    Latih WindowMLP dari data self-play (lihat generate_selfplay_windows).
    Data uji dibangkitkan dari game terpisah (seed berbeda) sehingga metrik
    mencerminkan sel perbatasan pada game yang belum pernah dilihat model.
    """
    cells, dens, mine = generate_selfplay_windows(n_games, rows, cols, mines, seed)
    t_cells, t_dens, t_mine = generate_selfplay_windows(
        n_test_games, rows, cols, mines, seed + 10_000_000)
    model = WindowMLP(hidden, onehot, use_density, seed=seed)
    if verbose:
        print(f"[selfplay] {n_games} game -> latih={len(cells):,} sampel "
              f"(bom {mine.mean():.1%}) | uji={len(t_cells):,} | "
              f"parameter={model.net.n_parameters:,}")
    Xte = model.pack(t_cells, t_dens)
    history = model.net.fit(model.pack(cells, dens), mine, epochs=epochs,
                            batch_size=batch_size, lr=lr, loss=loss,
                            optimizer=optimizer, seed=seed, val=(Xte, t_mine),
                            verbose=verbose, transform=model.transform)
    metrics = evaluate_binary(t_mine, model.net.predict(Xte, transform=model.transform))
    report = {"n_train": int(len(cells)), "n_test": int(len(t_cells)),
              "mine_rate_train": float(mine.mean()),
              "n_parameters": model.net.n_parameters, "test_metrics": metrics,
              "history": history,
              "config": {"games": n_games, "board": [rows, cols, mines],
                         "hidden": list(hidden), "onehot": onehot,
                         "use_density": use_density, "epochs": epochs}}
    return model, report


# --------------------------------------------------------------------------- #
# Solver: bermain dengan jendela geser (jurnal Bab 5.5.1 - 5.5.2)
# --------------------------------------------------------------------------- #
class NeuralWindowSolver:
    """
    Pelajar MLP yang memainkan permainan: jendela 5x5 digeser ke setiap sel
    perbatasan, MLP memberi P(bom), lalu sel dengan P(bom) terkecil dibuka.

    Murni jaringan saraf, tanpa SPS dan tanpa flag, mengikuti jurnal. Selalu
    mengembalikan sel tertutup selama masih ada, sehingga permainan tidak
    pernah macet (jurnal Bab 5.5.2).

    Antarmuka sama dengan BayesianMinesweeperSolver.recommend_move.
    """

    def __init__(self, model: WindowMLP) -> None:
        self.model = model
        self._helper = BayesianMinesweeperSolver()

    @classmethod
    def load(cls, path: str) -> "NeuralWindowSolver":
        return cls(WindowMLP.load(path))

    def recommend_move(self, game_state: np.ndarray, total_mines: int
                       ) -> Tuple[int, int, float, np.ndarray, str]:
        rows, cols = game_state.shape
        prob_matrix = np.full((rows, cols), np.nan, dtype=float)

        covered = [(int(r), int(c)) for r, c in np.argwhere(game_state == COVERED)]
        if not covered:
            return -1, -1, 1.0, prob_matrix, "NEURAL_NO_MOVE"

        bordering, _cons, _nb = self._helper.get_bordering_and_constraints(game_state)
        bordering = {cell for cell in bordering if game_state[cell] == COVERED}
        candidates = sorted(bordering) if bordering else covered
        reason = "NEURAL_BORDER" if bordering else "NEURAL_NO_BORDER"

        cells = window_cells(game_state, candidates)
        density = np.full(len(candidates), total_mines / float(rows * cols))
        probs = self.model.predict_mine(cells, density)

        for (r, c), p in zip(candidates, probs):
            prob_matrix[r, c] = float(p)

        order = sorted(
            range(len(candidates)),
            key=lambda i: (round(float(probs[i]), 9),
                           self._helper._tiebreak_key(candidates[i], rows, cols)))
        br, bc = candidates[order[0]]
        return br, bc, float(probs[order[0]]), prob_matrix, reason
