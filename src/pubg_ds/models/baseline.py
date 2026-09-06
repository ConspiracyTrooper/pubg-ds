"""Baseline, который финальная модель должна превзойти"""


import pandas   as pd
import lightgbm as lgb

from pubg_ds.features import schema
from pubg_ds.models   import predict


def lgb_baseline_mae(train: pd.DataFrame, val: pd.DataFrame, random_state: int) -> float:
    """MAE предсказания нетюненого lgb"""

    # Обучение lgb на train
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, random_state=random_state)
    model.fit(train[schema.MODEL_FEATURES], train[schema.TARGET])

    return predict.mae(model, val)
