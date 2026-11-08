"""
data_and_testing.py
===================

Modul ini menangani dua tanggung jawab utama dalam proyek Minesweeper:
1. Dataset Loading & Parsing (Manual string, CSV, Pandas DataFrame, JSON).
2. Automated Benchmarking (Membandingkan Baseline SPS+Random vs Solver Bayesian).

Cara pakai:
    python src/data_and_testing.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import komponen dari proyek kita
try:
    from .minesweeper_app import (MinesweeperGame, COVERED, FLAGGED, MINE,
                                  OUT_OF_BOARD, parse_board_text)
    from .bayesian_solver import BayesianMinesweeperSolver
except ImportError:  # dijalankan langsung: python src/data_and_testing.py
    from minesweeper_app import (MinesweeperGame, COVERED, FLAGGED, MINE,
                                 OUT_OF_BOARD, parse_board_text)
    from bayesian_solver import BayesianMinesweeperSolver

# Opsional: ms_toollib
try:
    import ms_toollib as ms
    HAS_MS_TOOLLIB = True
except ImportError:
    HAS_MS_TOOLLIB = False


# Kolom fitur dataset Kaggle: jendela 5x5 di sekitar sel target (sel tengah
# tidak ada). Urutan baris-demi-baris: cell_-2_-2, cell_-2_-1, ..., cell_2_2.
KAGGLE_CELL_COLUMNS: List[str] = [
    f"cell_{dr}_{dc}"
    for dr in range(-2, 3) for dc in range(-2, 3) if not (dr == 0 and dc == 0)
]


class DatasetLoader:
    """Kelas untuk mengurus input manual dan parsing dataset eksternal."""

    @staticmethod
    def from_manual_input(board_text: str, total_mines: Optional[int] = None) -> Dict[str, Any]:
        """
        Mengubah string grid/teks menjadi format standar NumPy.
        Simbol (sama dengan `parse_board_text`):
          - '?', '_', '-1' = tertutup      - 'F', '-3' = flag
          - '0'..'8'       = clue          - 'M', '*', '-4' = bom (ground truth)
          - '-2'           = di luar papan - '# mines=N'    = total bom
        `board_values` = None bila tidak ada simbol bom.
        """
        game_state, board_values, parsed_mines = parse_board_text(board_text)
        return {
            "board_values": board_values,
            "game_state": game_state,
            "total_mines": total_mines if total_mines is not None else parsed_mines,
        }

    @staticmethod
    def from_csv_or_hf(file_path_or_df: Union[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Parsing dataset tabular (Kaggle/HuggingFace).
        Mendukung format umum di mana board disimpan sebagai string JSON array
        atau dipisahkan spasi/koma.
        """
        if isinstance(file_path_or_df, str):
            if file_path_or_df.endswith('.csv'):
                df = pd.read_csv(file_path_or_df)
            elif file_path_or_df.endswith('.json'):
                df = pd.read_json(file_path_or_df)
            elif file_path_or_df.endswith('.xlsx'):
                return DatasetLoader.from_kaggle_dataset(file_path_or_df)
            else:
                raise ValueError("Format file harus .csv, .json, atau .xlsx")
        else:
            df = file_path_or_df

        # Format dataset Kaggle (kolom cell_R_C + global_density + safe)
        if all(col in df.columns for col in KAGGLE_CELL_COLUMNS):
            return DatasetLoader.from_kaggle_dataset(df)

        results = []
        for _, row in df.iterrows():
            # Cari kolom board (bisa 'board', 'state', 'matrix')
            board_str = row.get('board', row.get('state', row.get('matrix')))
            mines = row.get('mines', row.get('total_mines', None))
            
            if isinstance(board_str, str):
                try:
                    # Coba parse sebagai JSON array 2D
                    parsed = json.loads(board_str)
                    state = np.array(parsed, dtype=int)
                except json.JSONDecodeError:
                    # Atau parse sebagai raw text grid
                    parsed = DatasetLoader.from_manual_input(board_str)
                    state = parsed["game_state"]
                    if mines is None or pd.isna(mines):
                        mines = parsed["total_mines"]
            elif isinstance(board_str, list):
                state = np.array(board_str, dtype=int)
            else:
                continue
                
            if mines is None or pd.isna(mines):
                mines = max(int(np.sum(state == FLAGGED)), 1)
            results.append({
                "board_values": None,  # dataset hanya berisi state pemain
                "game_state": state,
                "total_mines": int(mines)
            })
            
        return results

    @staticmethod
    def _read_kaggle_frame(source: Union[str, pd.DataFrame],
                           nrows: Optional[int] = None) -> pd.DataFrame:
        """Baca CSV / XLSX (semua sheet 'Data*') / DataFrame dataset Kaggle."""
        if isinstance(source, str):
            if source.endswith('.xlsx'):
                xls = pd.ExcelFile(source)
                parts, got = [], 0
                for name in xls.sheet_names:
                    if not name.startswith('Data'):
                        continue
                    left = None if nrows is None else nrows - got
                    if left is not None and left <= 0:
                        break
                    part = xls.parse(name, nrows=left)
                    parts.append(part)
                    got += len(part)
                if not parts:
                    raise ValueError("Tidak ada sheet 'Data_*' di file xlsx.")
                return pd.concat(parts, ignore_index=True)
            if source.endswith('.csv'):
                return pd.read_csv(source, nrows=nrows)
            raise ValueError("Format file harus .csv atau .xlsx")
        return source if nrows is None else source.head(nrows)

    @staticmethod
    def kaggle_arrays(source: Union[str, pd.DataFrame],
                      nrows: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Versi vektor (cepat) untuk pelatihan model.

        Returns
        -------
        cells   : int array (n, 24), urutan = KAGGLE_CELL_COLUMNS
                  (-2 luar papan, -1 tertutup, 0..8 terbuka)
        density : float array (n,)  global_density (NaN bila kolom tidak ada)
        safe    : int array (n,)    label dataset, 1 = aman, 0 = bom (-1 bila tak ada)
        """
        df = DatasetLoader._read_kaggle_frame(source, nrows)
        missing = [c for c in KAGGLE_CELL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom dataset tidak lengkap, contoh: {missing[:3]}")
        cells = df[KAGGLE_CELL_COLUMNS].to_numpy(dtype=int)
        bad = set(np.unique(cells).tolist()) - set(range(-2, 9))
        if bad:
            raise ValueError(f"Nilai sel tidak dikenal untuk dataset: {sorted(bad)}")
        density = (df['global_density'].to_numpy(dtype=float)
                   if 'global_density' in df.columns else np.full(len(df), np.nan))
        safe = (df['safe'].to_numpy(dtype=int)
                if 'safe' in df.columns else np.full(len(df), -1))
        return cells, density, safe

    @staticmethod
    def from_kaggle_dataset(source: Union[str, pd.DataFrame],
                            nrows: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Memuat dataset Kaggle `minesweeper_dataset` (CSV, XLSX hasil rapi, atau
        DataFrame) TANPA konversi kode, karena encoding program sudah sama:
            -1 = tertutup, -2 = di luar papan, 0..8 = terbuka (dataset: 0..6).

        Tiap baris dataset adalah jendela 5x5 di sekitar satu sel target.
        Sel tengah (target) tidak ada di kolom dataset, jadi di `game_state`
        diisi COVERED (target memang belum diketahui pemain).

        Untuk .xlsx, semua sheet berawalan 'Data' dibaca berurutan (sheet 'Info'
        dilewati). Pakai `nrows` untuk membatasi jumlah baris karena file
        lengkap berisi 1,36 juta baris.

        Tiap elemen hasil:
            game_state     : np.ndarray 5x5
            board_values   : None (posisi bom asli tidak diketahui)
            safe           : label dataset (1 = aman, 0 = bom)
            global_density : kepadatan bom seluruh papan
            total_mines    : PERKIRAAN = round(global_density * jumlah sel dalam
                             papan di jendela), minimal 1. Ukuran papan asli
                             tidak ada di dataset, jadi ini hanya aproksimasi.
        """
        cells, density, labels = DatasetLoader.kaggle_arrays(source, nrows)

        # sisipkan sel target (indeks 12 pada jendela 5x5 yang diratakan)
        windows = np.insert(cells, 12, COVERED, axis=1).reshape(-1, 5, 5)

        results: List[Dict[str, Any]] = []
        for i in range(len(cells)):
            win = windows[i].copy()
            n_in_board = int(np.sum(win != OUT_OF_BOARD))
            mines = (max(1, int(round(density[i] * n_in_board)))
                     if not np.isnan(density[i]) else 1)
            results.append({
                "board_values": None,
                "game_state": win,
                "total_mines": mines,
                "safe": int(labels[i]),
                "global_density": float(density[i]),
            })
        return results

    @staticmethod
    def generate_sample_dataset(output_path: str = "minesweeper_test_dataset.csv",
                                seed: int = 42) -> pd.DataFrame:
        """
        Membuat 30 sampel board (10 Beginner, 10 Intermediate, 10 Expert)
        dan menyimpannya ke CSV.
        """
        presets = {
            "beginner": {"rows": 9, "cols": 9, "mines": 10},
            "intermediate": {"rows": 16, "cols": 16, "mines": 40},
            "expert": {"rows": 16, "cols": 30, "mines": 99}
        }
        
        data = []
        rng = np.random.default_rng(seed)
        
        for diff, config in presets.items():
            for i in range(10):
                # Generate board acak dengan MinesweeperGame
                game = MinesweeperGame(config["rows"], config["cols"], config["mines"])
                game.load_game(
                    start_row=int(rng.integers(0, config["rows"])),
                    start_col=int(rng.integers(0, config["cols"])),
                    rng=rng
                )
                
                # Simpan sebagai string JSON 2D
                state_str = json.dumps(game.game_state.tolist())
                data.append({
                    "id": f"{diff}_{i+1:02d}",
                    "difficulty": diff,
                    "rows": config["rows"],
                    "cols": config["cols"],
                    "mines": config["mines"],
                    "board": state_str
                })
                
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"[DatasetLoader] Berhasil membuat {len(df)} board tes di '{output_path}'")
        return df


class RandomGuessSolver:
    """
    Baseline Solver: SPS + Random Guess.
    Jika tidak ada langkah aman dari SPS, pilih sel COVERED secara acak.
    """
    def __init__(self, seed: Optional[int] = None) -> None:
        self._sps = BayesianMinesweeperSolver()
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def recommend_move(self, game_state: np.ndarray, total_mines: int) -> Tuple[int, int, float, np.ndarray, str]:
        empty = np.full(game_state.shape, np.nan)
        safe, _ = self._sps.apply_single_point_strategy(game_state)

        if safe:
            r, c = sorted(safe)[0]
            return r, c, 0.0, empty, "SPS_DETERMINISTIC"

        covered = np.argwhere(game_state == COVERED)
        if len(covered) == 0:
            return -1, -1, 1.0, empty, "RANDOM_GUESS"

        r, c = covered[int(self._rng.integers(len(covered)))]
        return int(r), int(c), float("nan"), empty, "RANDOM_GUESS"


class BenchmarkEvaluator:
    """Kelas untuk mengeksekusi evaluasi performa (Baseline vs Our Method)."""

    def __init__(self):
        self.results = []

    def _run_single_game(self, solver, difficulty: str, config: dict,
                         seed: Optional[int] = None, solver_label: str = "") -> Dict[str, Any]:
        """Mensimulasikan satu game dari awal sampai selesai."""
        game = MinesweeperGame(config["rows"], config["cols"], config["mines"])

        # Klik pertama di tengah; seed yang sama => papan yang sama untuk
        # semua solver (perbandingan adil).
        cr, cc = config["rows"] // 2, config["cols"] // 2
        game.load_game(cr, cc, rng=np.random.default_rng(seed))
        if hasattr(solver, "reseed") and seed is not None:
            solver.reseed(seed)
        
        start_time = time.time()
        moves = 0
        max_safety_moves = config["rows"] * config["cols"] * 2
        
        while game.game_status == "PLAYING" and moves < max_safety_moves:
            r, c, p, _, _ = solver.recommend_move(game.game_state, game.num_mines)
            if r < 0 or c < 0:
                break
                
            game.uncover(r, c)
            moves += 1
            
            # Auto-flag jika SPS memastikan itu bom (hanya untuk kerapian, 
            # tidak wajib untuk win rate)
            
        elapsed = time.time() - start_time
        
        return {
            "difficulty": difficulty,
            "solver": solver_label,
            "status": game.game_status,
            "won": 1 if game.game_status == "WON" else 0,
            "moves": moves,
            "cleared_pct": game.progress() * 100.0,
            "time_sec": elapsed
        }

    def run_benchmark(self, num_games_per_diff: int = 10, seed: int = 0,
                      output_csv: Optional[str] = "benchmark_results.csv",
                      extra_solvers: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Jalankan benchmark untuk semua tingkat kesulitan.

        extra_solvers: {label: solver} tambahan (mis. NeuralWindowSolver),
        dimainkan pada papan yang sama dengan solver lain.
        """
        configs = {
            "Beginner": {"rows": 9, "cols": 9, "mines": 10},
            "Intermediate": {"rows": 16, "cols": 16, "mines": 40},
            "Expert": {"rows": 16, "cols": 30, "mines": 99}
        }
        
        solvers = {
            "Baseline (Random)": RandomGuessSolver(),
            "Our Method (Bayesian)": BayesianMinesweeperSolver()
        }
        if extra_solvers:
            solvers.update(extra_solvers)
        
        self.results.clear()
        total_runs = len(configs) * len(solvers) * num_games_per_diff
        current_run = 0
        
        print(f"[Benchmark] Memulai {total_runs} simulasi game...")
        
        for diff_name, config in configs.items():
            for solver_name, solver in solvers.items():
                for i in range(num_games_per_diff):
                    current_run += 1
                    print(f"  [{current_run}/{total_runs}] {diff_name} | {solver_name} | Game {i+1}", end="\r")
                    
                    res = self._run_single_game(solver, diff_name, config,
                                                seed=seed + i,
                                                solver_label=solver_name)
                    self.results.append(res)
                    
        print("\n[Benchmark] Selesai!")
        
        # Agregasi data
        df = pd.DataFrame(self.results)
        summary = df.groupby(["difficulty", "solver"]).agg(
            win_rate=("won", lambda x: (x.sum() / len(x)) * 100),
            avg_moves=("moves", "mean"),
            avg_cleared=("cleared_pct", "mean"),
            avg_time=("time_sec", "mean")
        ).reset_index()
        
        if output_csv:
            summary.to_csv(output_csv, index=False)
            print(f"[Benchmark] Hasil disimpan ke '{output_csv}'")
        return summary

    def plot_benchmark_results(self, summary_df: pd.DataFrame, show: bool = True,
                               output_png: Optional[str] = "benchmark_visualization.png"):
        """Menghasilkan 3 grafik batang siap publikasi."""
        if summary_df.empty:
            print("DataFrame kosong. Jalankan run_benchmark() terlebih dahulu.")
            return
            
        # Style setup
        sns.set_theme(style="whitegrid", context="talk")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        palette = {"Baseline (Random)": "#E57373", "Our Method (Bayesian)": "#4CAF50"}
        extra_colors = ["#42A5F5", "#FFB74D", "#BA68C8", "#4DB6AC"]
        for name in summary_df["solver"].unique():
            if name not in palette:
                palette[name] = extra_colors[len(palette) % len(extra_colors)]
        
        # 1. Win Rate
        sns.barplot(
            data=summary_df, x="difficulty", y="win_rate", hue="solver",
            ax=axes[0], palette=palette, edgecolor="black"
        )
        axes[0].set_title("Win Rate (%)", fontweight="bold")
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_ylim(0, 105)
        axes[0].legend_.remove() # Hilangkan legenda di plot 1 & 2 agar rapi
        
        # 2. Average Moves
        sns.barplot(
            data=summary_df, x="difficulty", y="avg_moves", hue="solver",
            ax=axes[1], palette=palette, edgecolor="black"
        )
        axes[1].set_title("Average Moves Made", fontweight="bold")
        axes[1].set_ylabel("Moves")
        axes[1].legend_.remove()
        
        # 3. Execution Speed
        sns.barplot(
            data=summary_df, x="difficulty", y="avg_time", hue="solver",
            ax=axes[2], palette=palette, edgecolor="black"
        )
        axes[2].set_title("Execution Speed (Avg Sec/Game)", fontweight="bold")
        axes[2].set_ylabel("Seconds")
        axes[2].set_yscale("log") # Log scale karena Baseline sangat cepat vs Bayesian
        axes[2].legend(title="Solver", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        if output_png:
            plt.savefig(output_png, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point demonstrasi
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 70)
    print(" MINESWEEPER - DATA AND TESTING MODULE DEMO")
    print("=" * 70)
    
    # 1. Uji DatasetLoader (Manual)
    print("\n[1] Testing Manual Input...")
    sample_text = """
    ? ? ? ? 1 0 0 0 0
    ? ? ? ? 1 0 0 0 0
    ? ? ? ? 1 0 0 0 0
    ? ? ? ? 1 1 1 1 1
    ? ? ? ? ? ? ? ? ?
    """
    parsed = DatasetLoader.from_manual_input(sample_text, total_mines=5)
    print(f"    Game State Shape: {parsed['game_state'].shape}")
    print(f"    Total Mines: {parsed['total_mines']}")
    
    # 2. Generate Sample Dataset
    print("\n[2] Generating sample dataset...")
    df = DatasetLoader.generate_sample_dataset("minesweeper_test_dataset.csv")
    
    # 3. Load from Dataset
    print("\n[3] Loading from generated CSV...")
    loaded_boards = DatasetLoader.from_csv_or_hf("minesweeper_test_dataset.csv")
    print(f"    Berhasil memuat {len(loaded_boards)} boards dari CSV.")
    
    # 4. Jalankan Benchmark (Uji coba kecil: 3 game per tingkat)
    print("\n[4] Menjalankan Benchmark (3 games per difficulty)...")
    evaluator = BenchmarkEvaluator()
    summary = evaluator.run_benchmark(num_games_per_diff=3, seed=0)
    
    print("\n[5] Hasil Benchmark (Ringkasan):")
    print(summary.to_string(index=False))
    
    # 5. Plotting
    print("\n[6] Membuat grafik visualisasi...")
    evaluator.plot_benchmark_results(summary)
    print("\n[OK] Selesai! Cek folder untuk 'minesweeper_test_dataset.csv', 'benchmark_results.csv', dan 'benchmark_visualization.png'.")