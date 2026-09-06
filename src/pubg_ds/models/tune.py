"""Подбор гиперпараметров LightGBM через Optuna + GroupKFold по matchId"""


import json
import optuna
import numpy  as np
import pandas as pd

from pubg_ds                import config
from pubg_ds.features.split import group_kfold_by_match, split_by_match
from pubg_ds.models         import train, predict


# Одна точка гиперпараметров из конфигурируемого пространства
def suggest_params(trial: optuna.Trial, space: dict, n_estimators: int) -> dict:
    params = {'n_estimators': n_estimators}
    for name, s in space.items():
        if s['type'] == 'int':
            params[name] = trial.suggest_int(name, s['low'], s['high'], log=s.get('log', False))
        else:
            params[name] = trial.suggest_float(name, s['low'], s['high'], log=s.get('log', False))
    return params


# Итерация Optuna
def objective(trial: optuna.Trial, df: pd.DataFrame, space: dict, n_estimators: int, n_folds: int, seed: int) -> float:
    params = suggest_params(trial, space, n_estimators)

    # MAE по всем фолдам GroupKFold
    fold_maes = []
    for tr, val in group_kfold_by_match(df, n_folds):
        model = train.fit_lgb(tr, val, params, seed)
        fold_maes.append(predict.mae(model, val))

    return float(np.mean(fold_maes))


def main() -> None:
    p            = config.load_params()
    seed         = p['train']['random_state']
    n_folds      = p['train']['cv_folds']
    n_estimators = p['train']['n_estimators']
    n_trials     = p['tune']['n_trials']
    space        = p['tune']['search_space']

    # Матрица фич
    df = pd.read_parquet(config.DATA_PROCESSED / "reference.parquet")

    # Отрезаем тот же сплит, что и в train.py
    trainval, _ = split_by_match(df, p['train']['test_size'], seed)

    # Поиск с фиксированным сидом сэмплера
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(lambda t: objective(t, trainval, space, n_estimators, n_folds, seed), n_trials=n_trials)

    # Лучшие параметры + потолок n_estimators для финального fit_lgb
    best_params = {**study.best_params, 'n_estimators': n_estimators}

    config.MODELS_DIR.mkdir(exist_ok=True)
    with open(config.MODELS_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"best CV MAE = {study.best_value:.5f}")
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    main()
