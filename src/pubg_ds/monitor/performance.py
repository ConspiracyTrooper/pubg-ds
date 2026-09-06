"""Model degradation: live-MAE замороженной модели vs её MAE на Kaggle-holdout.

Shadow scoring с ground truth: API отдаёт завершённые матчи, поэтому предсказание
и настоящий winPlacePerc доступны одновременно, без лага.
"""

import json

import pandas as pd

from pubg_ds import config
from pubg_ds.models.predict import load_model, mae


# live-MAE frozen-модели, Kaggle-holdout MAE и величина деградации
def degradation_summary() -> dict:
    live   = pd.read_parquet(config.DATA_PROCESSED / "live.parquet")
    frozen = load_model(config.MODELS_DIR / "frozen_model.txt")

    # MAE на всём live-окне (со снапом к сетке maxPlace, как в predict)
    mae_live = mae(frozen, live)

    # Точка отсчёта — MAE той же модели на Kaggle-holdout (этап 4)
    with open(config.MODELS_DIR / "frozen_metrics.json") as f:
        mae_kaggle = json.load(f)["mae_frozen"]

    return {
        "mae_kaggle_holdout": mae_kaggle,
        "mae_live_frozen":    mae_live,
        "degradation_abs":    mae_live - mae_kaggle,
        "degradation_pct":    (mae_live - mae_kaggle) / mae_kaggle * 100,
        "n_live_rows":        len(live),
        "n_live_matches":     int(live["matchId"].nunique()),
    }


def main() -> None:
    print(json.dumps(degradation_summary(), indent=2))


if __name__ == "__main__":
    main()
