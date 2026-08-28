"""Построение матрицы признаков из train_V2.csv"""
import pandas as pd
from pubg_ds import config
from pubg_ds.features import schema


def build_reference(df: pd.DataFrame, match_type: str) -> pd.DataFrame:
    # Фильтр режима
    df = df[df['matchType'] == match_type]
    # Чистка NaN
    df = df.dropna(subset=[schema.TARGET])
    # Защита от битых строк 
    df = df[df['maxPlace'] > 1]
    # Приведение DataFrame к единой схеме
    df = df[schema.MATRIX_COLUMNS].reset_index(drop=True)
    return df


def main() -> None:
    params = config.load_params()
    # Чтение нужных колонок
    usecols = schema.MATRIX_COLUMNS + ['matchType']
    df = pd.read_csv(config.TRAIN_CSV, usecols=usecols)

    ref = build_reference(df, params['features']['match_type'])

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(config.DATA_PROCESSED / "reference.parquet", index=False)
    print(f'reference: {len(ref)} строк, {ref['matchId'].nunique()} матчей')

if __name__ == "__main__":
    main()