"""Переобучение на live-окне и сравнение MAE frozen vs retrained"""

import json

import mlflow
import pandas as pd

from pubg_ds import config
from pubg_ds.features.split import split_by_match
from pubg_ds.models import predict
from pubg_ds.models.train import fit_lgb


def main() -> None:
    p    = config.load_params()
    rp   = p["retrain"]
    seed = rp["random_state"]

    # Сплит live по matchId: train/val (early stopping) + test (сравнение)
    live = pd.read_parquet(config.DATA_PROCESSED / "live.parquet")
    trainval, test = split_by_match(live, rp["test_size"], seed)
    train, val     = split_by_match(trainval, rp["val_size"], seed)

    # Те же гиперпараметры, что у предыдущей модели
    with open(config.MODELS_DIR / "best_params.json") as f:
        best_params = json.load(f)

    with mlflow.start_run(run_name="retrained-live"):
        retrained = fit_lgb(train, val, best_params, seed)

        # Обе модели — на одном live-test
        frozen   = predict.load_model(config.MODELS_DIR / "frozen_model.txt")
        mae_froz = predict.mae(frozen, test)
        mae_retr = predict.mae(retrained, test)

        with open(config.MODELS_DIR / "frozen_metrics.json") as f:
            mae_kaggle = json.load(f)["mae_frozen"]

        comparison = {
            "mae_kaggle_test_frozen":  mae_kaggle,
            "mae_live_test_frozen":    mae_froz,
            "mae_live_test_retrained": mae_retr,
            "degradation_frozen":      mae_froz - mae_kaggle,   # цена дрифта на live
            "retrain_gain":            mae_froz - mae_retr,     # сколько отыграло переобучение
            "n_live_train_rows":       len(trainval),
            "n_live_test_rows":        len(test),
            "n_live_train_matches":    int(trainval["matchId"].nunique()),
            "n_live_test_matches":     int(test["matchId"].nunique()),
        }
        mlflow.log_params(best_params)
        mlflow.log_metrics({k: v for k, v in comparison.items() if isinstance(v, float)})

        config.MODELS_DIR.mkdir(exist_ok=True)
        retrained.booster_.save_model(str(config.MODELS_DIR / "retrained_model.txt"))
        mlflow.log_artifact(str(config.MODELS_DIR / "retrained_model.txt"))

    # Итоговая таблица трёх MAE — отчёт
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "mae_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
