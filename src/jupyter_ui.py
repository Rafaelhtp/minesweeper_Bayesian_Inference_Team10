"""
jupyter_ui.py
=============
Menampilkan UI pixel-art yang SAMA (web/index.html) langsung di dalam Jupyter Notebook.
Semua logika game + Bayesian tetap Python; notebook hanya menanam halaman via IFrame.

    from src.jupyter_ui import show
    show()                      # beginner; ganti ukuran lewat ikon gear di UI
"""
from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

try:
    from .web_app import Handler
except ImportError:
    from web_app import Handler

_SERVER = {}


def show(port: int = 8000, width: int = 1150, height: int = 1000):
    from IPython.display import IFrame, display
    if port not in _SERVER:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        _SERVER[port] = httpd
    return display(IFrame(f"http://127.0.0.1:{port}", width=width, height=height))
