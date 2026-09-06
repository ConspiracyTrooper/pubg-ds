import pandas as pd

from pubg_ds.features.split import split_by_match, group_kfold_by_match


# Синтетика: n_matches матчей по rows_per строк-игроков в каждом
def make_matches(n_matches=10, rows_per=4):
    rows = []
    for m in range(n_matches):
        for p in range(rows_per):
            rows.append({'matchId': f'm{m}', 'player': p})
    return pd.DataFrame(rows)


"""split_by_match"""

def test_no_match_in_both_sides():
    df = make_matches()
    rest, split = split_by_match(df, frac=0.3, seed=1)
    # матч целиком в одном наборе — id не пересекаются
    assert set(rest['matchId']).isdisjoint(set(split['matchId']))

def test_keeps_all_rows():
    df = make_matches()
    rest, split = split_by_match(df, frac=0.3, seed=1)
    assert len(rest) + len(split) == len(df)
    assert set(rest['matchId']) | set(split['matchId']) == set(df['matchId'])

def test_reproducible():
    df = make_matches()
    a = split_by_match(df, frac=0.3, seed=1)[1]['matchId'].tolist()
    b = split_by_match(df, frac=0.3, seed=1)[1]['matchId'].tolist()
    assert a == b


"""group_kfold_by_match"""

def test_fold_train_val_disjoint():
    df = make_matches()
    for train, val in group_kfold_by_match(df, n_splits=5):
        assert set(train['matchId']).isdisjoint(set(val['matchId']))

def test_each_match_once_as_val():
    df   = make_matches(n_matches=10)
    seen = []
    for _, val in group_kfold_by_match(df, n_splits=5):
        seen.extend(set(val['matchId']))
    # каждый matchId ровно один раз побывал в val
    assert sorted(seen) == sorted(set(df['matchId']))
