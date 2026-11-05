# Minesweeper Bayesian Solver

Solver Minesweeper hibrida: **Single Point Strategy (SPS)** deterministik +
**inferensi Bayesian eksak** untuk menghitung P(bom | evidence) tiap sel
tertutup (encoding mengikuti arXiv:2212.10446).

Encoding: `game_state` 0..8 = terungkap, `-2` = tertutup, `-3` = flag;
`board_values` -1 = bom, 0..8 = jumlah bom tetangga.

## Struktur
```
main.py                  CLI: play | solve | benchmark
src/bayesian_solver.py   SPS + enumerasi eksak per komponen + konvolusi
src/minesweeper_app.py   engine game, parser board, plot, UI ipywidgets, CLI
src/data_and_testing.py  DatasetLoader, baseline random, benchmark
tests/test_solver.py     unit test (termasuk validasi brute force)
examples/                contoh board manual
```

## Pemakaian
```bash
pip install -r requirements.txt
python main.py play --difficulty beginner --seed 1
python main.py solve --board examples/sample_board.txt
python main.py benchmark --games 5 --seed 0 --no-plot
python -m unittest discover -s tests -v
```

## Format board manual
`?`/`_`/`-2` tertutup · `F`/`-3` flag · `0..8` clue · `M`/`*`/`-1` bom
(ground truth, opsional) · baris `# mines=N` untuk total bom.

## Cara kerja Bayesian
Sel perbatasan dipecah jadi komponen independen. Tiap komponen dienumerasi
(backtracking) dan hanya menyimpan, per jumlah bom `k`, banyaknya solusi dan
berapa kali tiap sel jadi bom. Komponen digabung lewat konvolusi dan diberi
bobot `C(n_interior, sisa_bom - k)` (prior jumlah bom global), dengan aritmetika
integer eksak. Jika sebuah komponen terlalu besar (>1,5 juta node), dipakai
estimasi lokal sebagai fallback.
