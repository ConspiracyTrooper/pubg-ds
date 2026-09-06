"""Обучение финальной модели"""


import json
import mlflow
import pandas   as pd
import lightgbm as lgb

from pubg_ds                import config
from pubg_ds.features       import schema
from pubg_ds.features.split import split_by_match
from pubg_ds.models         import baseline, predict


# Обучение lgb c остановкой по val
def fit_lgb(train: pd.DataFrame, val: pd.DataFrame, params: dict, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params, random_state=seed, verbose=-1, n_jobs=-1)
    model.fit(
        train[schema.MODEL_FEATURES], train[schema.TARGET],
        eval_X=val[schema.MODEL_FEATURES], eval_y=val[schema.TARGET],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return model


def main() -> None:
    p = config.load_params()
    seed = p['train']['random_state']

    # Бэкенд MLflow берётся из env MLFLOW_TRACKING_URI (см. Makefile / .env)

    # Лучшие гиперпараметры с этапа tune
    with open(config.MODELS_DIR / "best_params.json") as f:
        best_params = json.load(f)

    # Матрица фич
    df = pd.read_parquet(config.DATA_PROCESSED / "reference.parquet")

    # train / val / test split по matchId
    trainval, test = split_by_match(df,       p['train']['test_size'], seed)
    train,    val  = split_by_match(trainval, p['train']['val_size'],  seed)

    # Финальная модель, baseline
    model = fit_lgb(train, val, best_params, seed)
    mae_frozen   = predict.mae(model, test)
    mae_baseline = baseline.lgb_baseline_mae(trainval, test, seed)

    # Артефакт + метрики
    config.MODELS_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(config.MODELS_DIR / "frozen_model.txt"))
    metrics = {
        'mae_frozen':     mae_frozen,
        'mae_baseline':   mae_baseline,
        'best_iteration': model.best_iteration_,
    }
    with open(config.MODELS_DIR / "frozen_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Лог в MLflow
    with mlflow.start_run(run_name="frozen"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(config.MODELS_DIR / "frozen_model.txt"))
        


if __name__ == "__main__":
    main()
