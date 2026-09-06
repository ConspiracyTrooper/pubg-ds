"""Инференс. предсказание со снапом к сетке, метрика, загрузка модели"""

from pathlib import Path
import lightgbm as lgb
import numpy    as np
import pandas   as pd

from sklearn.metrics  import mean_absolute_error
from pubg_ds.features import schema


# Снап к сетке таргета
def _snap_to_grid(pred: np.ndarray, max_place: np.ndarray) -> np.ndarray:
    steps = np.maximum(np.asarray(max_place, dtype=float) - 1.0, 1.0)
    pred  = np.clip(pred, 0.0, 1.0)
    return np.round(pred * steps) / steps


# Предсказание обученной lgb-модели
def predict(model, df: pd.DataFrame) -> np.ndarray:
    raw = model.predict(df[schema.MODEL_FEATURES])
    return _snap_to_grid(raw, df['maxPlace'].to_numpy())


# MAE предсказания
def mae(model, df: pd.DataFrame) -> float:
    return float(mean_absolute_error(df[schema.TARGET], predict(model, df)))


# Загрузка модели из артефакта
def load_model(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
