# Minesweeper: Prediksi Bom dengan Bayesian Inference

**Proyek 1 Kecerdasan Buatan · Topik: Uncertainty Factor · Kelompok 10**
Program Studi S1 Informatika, Fakultas Informatika dan Teknik Elektro, Institut Teknologi Del

Minesweeper adalah permainan dengan informasi tak lengkap: angka pada sel yang terbuka hanya
menyatakan *berapa* bom di sekitarnya, bukan *di mana*. Proyek ini memakai **Teorema Bayes**
untuk menyatakan ketidakpastian itu sebagai peluang bom per sel, lalu memilih sel dengan
peluang bom terkecil.

Implementasi pada repositori ini adalah solver hibrida: **Single Point Strategy (SPS)**
deterministik + **inferensi Bayesian eksak** untuk menghitung P(bom | evidence) tiap sel
tertutup (encoding mengikuti arXiv:2212.10446), dilengkapi antarmuka web pixel-art yang
menampilkan peluang bom dan rincian perhitungannya.

## Fitur
- Solver hibrida: SPS untuk langkah pasti, inferensi Bayesian eksak untuk tebakan terbaik.
- Tiga tingkat kesulitan: Beginner 9x9 (10 bom), Intermediate 16x16 (40 bom), Expert 16x30 (99 bom).
- Klik pertama selalu aman (bom diletakkan setelah klik pertama).
- Antarmuka web: peluang bom (%) per sel, panel hitungan Bayes di samping papan,
  petunjuk AI, auto-solve dengan tombol jeda/lanjut.
- CLI untuk bermain, meminta rekomendasi dari board manual, dan benchmark
  (baseline random vs Bayesian).

## Struktur
```
main.py                  CLI: play | solve | benchmark
src/bayesian_solver.py   SPS + enumerasi eksak per komponen + konvolusi
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
`game_state`: 0..8 = terungkap, `-2` = tertutup, `-3` = flag.
`board_values`: -1 = bom, 0..8 = jumlah bom tetangga.

> Catatan: encoding ini mengikuti arXiv:2212.10446 dan **berbeda** dari dataset Kaggle
> (di mana kode -1 dan -2 diduga berarti "tertutup" dan "di luar papan"). Dataset Kaggle belum
> dipakai langsung oleh kode ini; bila dipakai, kodenya perlu dikonversi lebih dulu.

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
Jalankan dari root proyek. Jika tampilan tidak berubah setelah update, tekan Ctrl+F5.

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
python examples/example_manual_board.py
```

### Unit test
```bash
python -m unittest discover -s tests -v
```

## Format board manual
`?`/`_`/`-2` tertutup · `F`/`-3` flag · `0..8` clue · `M`/`*`/`-1` bom
(ground truth, opsional) · baris `# mines=N` untuk total bom.

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

## Evaluasi
`python main.py benchmark` membandingkan solver Bayesian dengan baseline random pada tiga
tingkat kesulitan. Sebagai konteks, benchmark solver Hugging Face (1.000 permainan per tingkat,
klik pertama dijamin aman) melaporkan win rate 95,5% (Pemula 9x9, 10 bom), 85,6% (Menengah
16x16, 40 bom), dan 36,2% (Ahli 30x16, 99 bom). Angka itu tidak setara langsung dengan
benchmark di repositori ini karena pengaturannya berbeda.

## Lisensi
Lihat berkas [LICENSE](LICENSE).