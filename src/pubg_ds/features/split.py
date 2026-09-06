"""Сплит по matchId"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from typing import Iterator


# Разделение по матчам
def split_by_match(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    rest_idx, split_idx = next(splitter.split(df, groups=df['matchId']))
    return df.iloc[rest_idx], df.iloc[split_idx]


# Фолды (train, val) для Optuna
def group_kfold_by_match(df: pd.DataFrame, n_splits: int) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    kfold = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in kfold.split(df, groups=df['matchId']):
        yield df.iloc[train_idx], df.iloc[val_idx]