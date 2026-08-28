import numpy as np
import pandas as pd

from pubg_ds.features import schema
from pubg_ds.features.reference import build_reference


def make_row(**over):
    row = {c: 0 for c in schema.FEATURES}
    row.update({
        'matchId': 'm1',
        'groupId': 'g1',
        'maxPlace': 28,
        'matchType': 'squad-fpp',
        schema.TARGET: 0.5,
    })
    row.update(over)
    return row


"""Состав схемы"""

def test_excluded_absent_from_matrix():
    assert set(schema.EXCLUDED).isdisjoint(schema.MATRIX_COLUMNS)

def test_build_reference_drops_leakage_columns():
    df  = pd.DataFrame([make_row(killPlace=5, rankPoints=1200)])
    out = build_reference(df, "squad-fpp")
    assert "killPlace"  not in out.columns
    assert "rankPoints" not in out.columns


"""Логика build_reference"""

def test_filters_match_type():
    df = pd.DataFrame([
        make_row(matchId='a'),
        make_row(matchId='b', matchType='solo-fpp')    # другой режим
    ])
    out = build_reference(df, 'squad-fpp')
    assert len(out) == 1
    assert out.iloc[0]['matchId'] == 'a'

def test_drops_nan_target():
    df = pd.DataFrame([
        make_row(matchId="a"),
        make_row(matchId="b", winPlacePerc=np.nan),    # NaN-таргет
    ])
    assert len(build_reference(df, "squad-fpp")) == 1

def test_drops_maxplace_1():
    df = pd.DataFrame([
        make_row(matchId="a", maxPlace=28),
        make_row(matchId="b", maxPlace=1),             # битая строка
    ])
    assert len(build_reference(df, "squad-fpp")) == 1

def test_output_columns_exact_order():
    df = pd.DataFrame([make_row()])
    out = build_reference(df, "squad-fpp")
    # состав и порядок колонок = MATRIX_COLUMNS; matchType не попал в матрицу
    assert list(out.columns) == schema.MATRIX_COLUMNS
    assert "matchType" not in out.columns