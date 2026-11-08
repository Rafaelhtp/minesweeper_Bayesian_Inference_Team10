# Minesweeper: Prediksi Bom dengan Bayesian Inference

**Proyek 1 Kecerdasan Buatan · Topik: Uncertainty Factor · Kelompok 10**
Program Studi S1 Informatika, Fakultas Informatika dan Teknik Elektro, Institut Teknologi Del

Minesweeper adalah permainan dengan informasi tak lengkap: angka pada sel yang terbuka hanya
menyatakan *berapa* bom di sekitarnya, bukan *di mana*. Proyek ini memakai **Teorema Bayes**
untuk menyatakan ketidakpastian itu sebagai peluang bom per sel, lalu memilih sel dengan
peluang bom terkecil.

Implementasi pada repositori ini adalah solver hibrida: **Single Point Strategy (SPS)**
deterministik + **inferensi Bayesian eksak** untuk menghitung P(bom | evidence) tiap sel
tertutup (encoding disamakan dengan dataset Kaggle), dilengkapi antarmuka web pixel-art yang
menampilkan peluang bom dan rincian perhitungannya.

## Fitur
- Solver hibrida: SPS untuk langkah pasti, inferensi Bayesian eksak untuk tebakan terbaik.
- Tiga tingkat kesulitan: Beginner 9x9 (10 bom), Intermediate 16x16 (40 bom), Expert 16x30 (99 bom).
- Klik pertama selalu aman (bom diletakkan setelah klik pertama). Sesuai versi final jurnal,
  hanya sel yang diklik yang dijamin aman dan boleh bernilai angka (`first_click="safe"`,
  default); mode `"zero"` (klik pertama pasti 0) tersedia sebagai opsi.
- Pelajar MLP jendela 5x5 sesuai jurnal arXiv:2212.10446 (numpy murni), dapat dilatih dari
  dataset Kaggle maupun dari data self-play, lalu dibandingkan dengan solver Bayesian.
- Antarmuka web: peluang bom (%) per sel, panel hitungan Bayes di samping papan,
  petunjuk AI, auto-solve dengan tombol jeda/lanjut.
- CLI untuk bermain, meminta rekomendasi dari board manual, dan benchmark
  (baseline random vs Bayesian).

## Struktur
```
main.py                  CLI: play | solve | benchmark | train
src/bayesian_solver.py   SPS + enumerasi eksak per komponen + konvolusi
src/neural_learner.py    MLP jendela 5x5 (jurnal Bab 5.5): pelatihan + solver saraf
models/                  model MLP terlatih (.npz)
src/minesweeper_app.py   engine game, parser board, plot, UI ipywidgets, CLI
src/data_and_testing.py  DatasetLoader, baseline random, benchmark
src/web_app.py           server web lokal (hanya library standar Python)
src/explain.py           rincian langkah perhitungan Bayes untuk UI web
src/jupyter_ui.py        menampilkan UI web di dalam Jupyter Notebook
web/index.html           antarmuka web (pixel-art)
tests/test_solver.py     unit test (termasuk validasi brute force)
examples/                contoh board manual
```

## Encoding
Encoding program **disamakan dengan dataset Kaggle** (`minesweeper_dataset`), sehingga baris
dataset bisa dipakai langsung tanpa konversi kode.

| Makna | Dataset Kaggle | Program |
|---|---|---|
| Sel tertutup | `-1` | `-1` (`COVERED`) |
| Di luar papan | `-2` | `-2` (`OUT_OF_BOARD`) |
| Sel terbuka (jumlah ranjau di sekitarnya) | `0..6` (dataset) | `0..8` |
| Bendera | tidak ada | `-3` (`FLAGGED`) |
| Bom (ground truth / bom yang meledak) | tidak ada | `-4` (`MINE`, kode internal) |

- `game_state` memakai `-1`, `-2`, `-3`, dan `0..8`.
- `board_values` (ground truth) memakai `-4` = bom dan `0..8` = jumlah bom tetangga.
- Sel `-2` tidak pernah dianggap tertutup, clue, atau flag, sehingga solver mengabaikannya.
  Ini yang membuat jendela 5x5 di dataset (yang memuat sel di luar papan) aman diproses.
- Konstanta didefinisikan satu kali di `src/bayesian_solver.py`; modul lain meng-import-nya,
  dan antarmuka web menerima kodenya dari server (`codes` pada `/api/state`).
- Kode lama (tertutup `-2`, bom `-1`) **tidak lagi berlaku**. File board atau CSV yang dibuat
  dengan encoding lama harus diubah: `-2` -> `-1` untuk sel tertutup, dan `-1` -> `-4` untuk bom.

### Memuat dataset Kaggle
```python
from src.data_and_testing import DatasetLoader
items = DatasetLoader.from_kaggle_dataset("minesweeper_dataset.csv", nrows=1000)
# atau file xlsx hasil rapi (sheet Data_1, Data_2, ...; sheet Info dilewati)
items = DatasetLoader.from_kaggle_dataset("minesweeper_dataset_terbaru.xlsx", nrows=1000)
item = items[0]   # game_state 5x5 (sel tengah = target, diisi -1), safe, global_density
```
Catatan: dataset hanya berisi jendela 5x5 dan `global_density`, bukan ukuran papan. Karena itu
`total_mines` hasil loader adalah **perkiraan** (`global_density` x jumlah sel dalam papan di
jendela), bukan nilai asli.

## Instalasi
```bash
pip install -r requirements.txt
```

## Pemakaian

### Antarmuka web
```bash
python src/web_app.py            # lalu buka http://127.0.0.1:8000
python src/web_app.py --port 9000 --no-browser
```
- Jalankan dari root proyek.
- **Klik kiri** membuka sel, **klik kanan** (atau tekan lama di layar sentuh) memasang flag.
- Tiap sel tertutup menampilkan peluang bom dalam persen (angka kecil di pojok kanan bawah;
  **merah 100** = pasti bom). Panel di samping papan menampilkan rincian perhitungan Bayes
  untuk sel yang ditunjuk kursor.
- Tombol di bagian atas:

| Ikon | Fungsi |
|---|---|
| PETUNJUK | AI menyarankan langkah teraman |
| JEDA / LANJUT | menjeda atau melanjutkan auto-solve (muncul saat auto-solve aktif) |
| WAKTU | pewaktu permainan |
| ULANG | mulai game baru |
| SISA BOM | jumlah bom dikurangi flag |
| MENU | tingkat kesulitan, AUTO-SOLVE, tampil % bom |

- Setelah menang atau kalah, tekan **R** atau **Enter** untuk main lagi.

### Command line
```bash
python main.py play --difficulty beginner --seed 1
python main.py solve --board examples/sample_board.txt
python main.py benchmark --games 5 --seed 0 --no-plot
python main.py benchmark --games 20 --no-plot --neural models/mlp_selfplay_beginner.npz
python main.py train --source selfplay --games 6000 --epochs 12 --out models/mlp_selfplay_beginner.npz
python main.py train --source dataset --data minesweeper_dataset_terbaru.xlsx --out models/mlp_dataset.npz
python examples/example_manual_board.py
```

### Unit test
```bash
python -m unittest discover -s tests -v
```

## Format board manual
`?`/`_`/`-1` tertutup · `-2` di luar papan · `F`/`-3` flag · `0..8` clue ·
`M`/`*`/`-4` bom (ground truth, opsional) · baris `# mines=N` untuk total bom.

## Cara kerja
Sel perbatasan dipecah jadi komponen independen. Tiap komponen dienumerasi
(backtracking) dan hanya menyimpan, per jumlah bom `k`, banyaknya solusi dan
berapa kali tiap sel jadi bom. Komponen digabung lewat konvolusi dan diberi
bobot `C(n_interior, sisa_bom - k)` (prior jumlah bom global), dengan aritmetika
integer eksak. Jika sebuah komponen terlalu besar (>1,5 juta node), dipakai
estimasi lokal sebagai fallback.

Pemilihan langkah:
1. Jalankan SPS. Jika ada sel yang pasti aman, buka sel itu.
2. Jika tidak ada, hitung P(bom | evidence) semua sel tertutup dan buka sel dengan peluang
   terkecil (seri: sudut, lalu tepi, lalu bagian dalam).

## Pelajar MLP (jurnal arXiv:2212.10446 + dataset Kaggle)
Kesesuaian antara jurnal, dataset, dan kode:

| Jurnal | Dataset | Kode |
|---|---|---|
| Jendela geser 5x5, 24 input, sel tengah = target (Bab 5.5.1) | 24 kolom `cell_R_C` | `window_cells()` (urutan sama dengan kolom dataset) |
| Keluaran 0..1 = peluang bom, pilih yang terkecil (Bab 5.5.2) | label `safe` (1 = aman) | dilatih sebagai P(bom) = 1 - safe |
| MLP ReLU + sigmoid, MSE atau cross-entropy, 70-80-60-35 (Bab 5.5.3) | - | `NumpyMLP`, `DEFAULT_HIDDEN` |
| Hanya sel perbatasan yang dinilai, learner tidak boleh flag (Bab 3.2, 5.5.1) | dataset tanpa flag | `NeuralWindowSolver` |
| Data latih dibangkitkan saat game dimainkan (Bab 5.5.4) | - | `generate_selfplay_windows()` |
| Klik pertama hanya dijamin aman (Bab 5.6.2.1) | - | `load_game(first_click="safe")` |
| One-hot untuk input kategorikal (Bab 5.6.1.2, untuk CNN) | - | opsi `onehot=True` pada MLP |

Catatan encoding: jurnal memakai tertutup `-2` dan bom `-1`, sedangkan dataset memakai
tertutup `-1` dan luar papan `-2`. Kode mengikuti **dataset** (lihat bagian Encoding).
CNN dari jurnal (Bab 5.6) tidak diimplementasikan; hanya MLP yang sesuai dengan format dataset.

### Dua sumber data latih
1. **Dataset Kaggle** (`--source dataset`). Duplikat dibuang sebelum membagi data latih dan
   validasi. Baris dataset **tidak diacak**: ~300 ribu baris pertama hanya 62,7% aman, sisanya
   ~85%. Karena itu `--rows N` memberi sampel bias; biarkan default (semua baris).
2. **Self-play** (`--source selfplay`), cara jurnal Bab 5.5.4: jendela dari setiap sel perbatasan
   dicatat sepanjang game, label = posisi bom sebenarnya.

Mengapa model dari dataset kalah saat bermain: pada dataset hanya 19,6% target yang bom, sedangkan
di antara sel perbatasan pada game nyata ~51% adalah bom (dan jendela game nyata memuat lebih
banyak petunjuk: rata-rata 9,7 sel terbuka vs 6,4 pada dataset). Model belajar membedakan bom
dan aman pada distribusi dataset (AUC validasi 0,82), tetapi peringkat dan kalibrasinya tidak
cocok untuk memilih sel teraman saat bermain.

### Hasil (klik pertama aman, papan dan seed sama untuk semua solver)
Win rate dengan interval kepercayaan 95% (Wilson). N = 200 / 100 / 50 game
(Beginner / Intermediate / Expert).

| Solver | Beginner | Intermediate | Expert |
|---|---|---|---|
| Random + SPS (baseline) | 0,0% | 0,0% | 0,0% |
| **Bayesian** | **86,5%** (81,1-90,6) | **68,0%** (58,3-76,3) | **34,0%** (22,4-47,8) |
| MLP dari dataset, one-hot | 3,5% (1,7-7,0) | 0,0% | 0,0% |
| MLP dari dataset, 24 input | 1,5% (0,5-4,3) | 0,0% | 0,0% |
| MLP dari self-play (beginner) | 21,0% (15,9-27,2) | 0,0% | 0,0% |

Metrik prediksi (data validasi/uji terpisah):

| Model | Data uji | Akurasi | AUC |
|---|---|---|---|
| MLP dataset, one-hot | validasi dataset (106.992 baris unik) | 82,2% (baseline mayoritas 79,7%) | 0,816 |
| MLP dataset, 24 input | validasi dataset | 81,9% | 0,805 |
| MLP self-play | sel perbatasan 300 game terpisah | 90,2% | 0,963 |

Sebagai pembanding, jurnal melaporkan solver CSP 91,3 / 75,8 / 32,8% dan MLP terbaiknya
~40% pada Beginner (dilatih online pada ratusan ribu game); model self-play di sini hanya
dilatih dari 6.000 game, sehingga angkanya lebih rendah. Model Beginner tidak
digeneralisasi ke papan yang lebih besar. Latih terpisah per tingkat dengan
`--difficulty intermediate|expert` (jurnal juga melatih learner per mode).

## Evaluasi
`python main.py benchmark` membandingkan solver Bayesian dengan baseline random pada tiga
tingkat kesulitan. Sebagai konteks, benchmark solver Hugging Face (1.000 permainan per tingkat,
klik pertama dijamin aman) melaporkan win rate 95,5% (Pemula 9x9, 10 bom), 85,6% (Menengah
16x16, 40 bom), dan 36,2% (Ahli 30x16, 99 bom). Angka itu tidak setara langsung dengan
benchmark di repositori ini karena pengaturannya berbeda.

## Lisensi
Lihat berkas [LICENSE](LICENSE).