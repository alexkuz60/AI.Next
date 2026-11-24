from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv_data"
DEFAULT_CSV = CSV_DIR / "4x20.csv"

APP_DEBUG = os.getenv("APP_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8050"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "16"))

STARTUP_ORDER = 0
STARTUP_HIST_BACK = 1
STARTUP_MATRIX_SIZE = 200
STARTUP_DEEP = 0
STARTUP_DEEP_MODE = 0
STARTUP_SORTED = False
