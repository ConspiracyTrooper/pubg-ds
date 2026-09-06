"""Фичи из сырых json матчей PUBG API — parity с reference-схемой.

Зеркалит reference.py: сырая матрица в MATRIX_COLUMNS -> тот же
add_match_features -> FULL_COLUMNS. Один код derived => схемы совпадают.

Participant-stats API маппятся на Kaggle-фичи 1:1 (имена совпадают), кроме
matchDuration (из атрибутов матча) и maxPlace/groupId (из ростеров). Таргет —
той же формулой, что в Kaggle: winPlacePerc = (maxPlace − winPlace)/(maxPlace − 1).
"""

import json

import pandas as pd

from pubg_ds import config
from pubg_ds.features import schema
from pubg_ds.features.derive import add_match_features

# Сырые фичи из participant stats (имена совпадают с Kaggle); matchDuration — из матча
DIRECT_STATS = [f for f in schema.FEATURES if f != "matchDuration"]


# Разворачивает json одного матча в строки единой схемы (по игроку)
def match_to_rows(match: dict) -> list[dict]:
    data     = match["data"]
    match_id = data["id"]
    duration = data["attributes"]["duration"]

    # Раскладываем included по типам
    participants = {}
    rosters      = []
    for inc in match.get("included", []):
        if inc["type"] == "participant":
            participants[inc["id"]] = inc["attributes"]["stats"]
        elif inc["type"] == "roster":
            rosters.append(inc)

    # Структура матча из финального ростера (parity с maxPlace Kaggle)
    if not rosters:
        return []
    max_place = max(r["attributes"]["stats"]["rank"] for r in rosters)
    # Битые/вырожденные матчи (как фильтр maxPlace > 1 в reference)
    if max_place <= 1:
        return []

    rows = []
    for roster in rosters:
        group_id = roster["id"]
        for ref in roster["relationships"]["participants"]["data"]:
            stats = participants.get(ref["id"])
            if stats is None:
                continue
            row = {f: stats[f] for f in DIRECT_STATS}
            row["matchId"]       = match_id
            row["groupId"]       = group_id
            row["maxPlace"]      = max_place
            row["matchDuration"] = duration
            # Таргет — та же формула, что в Kaggle (parity по таргету)
            row[schema.TARGET]   = (max_place - stats["winPlace"]) / (max_place - 1)
            rows.append(row)
    return rows


# Собирает сырую live-матрицу в единой схеме из списка json-файлов матчей
def build_live(match_files: list) -> pd.DataFrame:
    rows = []
    for path in match_files:
        rows += match_to_rows(json.loads(path.read_text()))
    df = pd.DataFrame(rows)
    return df[schema.MATRIX_COLUMNS]


def main() -> None:
    files = sorted(config.DATA_LIVE.glob("*.json"))
    raw   = build_live(files)

    # Производные фичи материализуем прямо в матрицу (как в reference)
    live = add_match_features(raw)[schema.FULL_COLUMNS]

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    live.to_parquet(config.DATA_PROCESSED / "live.parquet", index=False)
    print(f'live: {len(live):,} строк, {live["matchId"].nunique():,} матчей, {len(schema.MODEL_FEATURES)} фич')


if __name__ == "__main__":
    main()
