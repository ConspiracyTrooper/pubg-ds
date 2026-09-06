"""Производные match-relative фичи поверх единой схемы.

Один и тот же код гоняется на reference (Kaggle) и live (API) — parity
сохраняется автоматически. Ранги внутри матча берут нейтральный тай-брейк
(average) и НЕ кодируют порядок выбывания, в отличие от исключённого killPlace.
"""

import pandas as pd

from pubg_ds.features import schema


# Добавляет ранги/нормировки внутри матча и агрегаты по группе (без заглядывания в таргет)
def add_match_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g  = df.groupby("matchId")

    # Процентильный ранг стата внутри матча
    for stat in schema.RANK_STATS:
        df[f"{stat}MatchRank"] = g[stat].rank(pct=True)

    # Нормировка на среднее по матчу — насколько игрок выделяется на фоне лобби
    for stat in schema.NORM_STATS:
        df[f"{stat}MatchNorm"] = df[stat] / (g[stat].transform("mean") + 1e-9)

    # Групповые агрегаты и ранг группы внутри матча — тиммейты делят одно место
    gg = df.groupby(["matchId", "groupId"])
    df["groupSize"] = gg["matchId"].transform("size")
    for stat in schema.GROUP_STATS:
        grp_mean = gg[stat].transform("mean")
        df[f"{stat}GroupMean"] = grp_mean
        df[f"{stat}GroupRank"] = grp_mean.groupby(df["matchId"]).rank(pct=True)

    return df
