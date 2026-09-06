"""Схема признаков — источник правды для reference (Kaggle) и live (API)"""

# Сырые фичи: маппятся 1:1 Kaggle ↔ PUBG API. Кормят derived + часть — фичи модели.
FEATURES   = ['walkDistance', 'rideDistance', 'kills', 'damageDealt', 'boosts', 'heals',
              'weaponsAcquired', 'longestKill', 'DBNOs', 'killStreaks', 'matchDuration']
TARGET     = "winPlacePerc"
ID_COLUMNS = ['matchId', 'groupId', 'maxPlace']         # не фичи, нужны для сплита/derive/снапа
EXCLUDED   = ['killPlace', 'rankPoints', 'killPoints', 'winPoints', 'numGroups',
              'assists', 'headshotKills', 'revives', 'roadKills',
              'swimDistance', 'teamKills', 'vehicleDestroys']   # утечка, легаси, слабые по важности

# Производные match-relative фичи (считаются в derive.add_match_features)
RANK_STATS  = ['walkDistance', 'rideDistance']                          # процентильный ранг в матче
NORM_STATS  = ['walkDistance']                                          # нормировка на среднее по матчу
GROUP_STATS = ['walkDistance', 'kills', 'damageDealt',                  # агрегаты по группе
               'boosts', 'weaponsAcquired', 'longestKill']

DERIVED = ([f"{s}MatchRank" for s in RANK_STATS]
           + [f"{s}MatchNorm" for s in NORM_STATS]
           + [f"{s}GroupMean" for s in GROUP_STATS]
           + [f"{s}GroupRank" for s in GROUP_STATS]
           + ['groupSize'])

# Фичи модели: сырые + производные (src везде обучается на этом наборе)
MODEL_FEATURES = FEATURES + DERIVED

MATRIX_COLUMNS = ID_COLUMNS + FEATURES + [TARGET]           # сырая матрица (из csv / build_reference)
FULL_COLUMNS   = ID_COLUMNS + MODEL_FEATURES + [TARGET]     # матрица в parquet (после derive)
