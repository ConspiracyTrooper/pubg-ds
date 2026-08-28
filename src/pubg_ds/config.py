"""Пути и гиперпараметры"""

from pathlib import Path
import yaml

# Корень проекта
PROJECT_ROOT   = Path(__file__).resolve().parents[2]

# Пути к данным
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV      = DATA_RAW / "train_V2.csv"

# Гиперпараметры и пороги
def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)