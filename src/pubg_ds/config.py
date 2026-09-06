"""Пути и гиперпараметры"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Корень проекта
PROJECT_ROOT   = Path(__file__).resolve().parents[2]

# Загрузка .env
load_dotenv(PROJECT_ROOT / ".env")

# Пути к данным
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_LIVE      = DATA_RAW / "live"                    # сырые json live-матчей
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV      = DATA_RAW / "train_V2.csv"
MODELS_DIR     = PROJECT_ROOT / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"

# PUBG API
PUBG_API_KEY   = os.getenv("PUBG_API_KEY", "")
PUBG_API_BASE  = "https://api.pubg.com"

# Гиперпараметры и пороги
def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)