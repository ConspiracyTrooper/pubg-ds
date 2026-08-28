"""Схема признаков, используемых в модели"""

FEATURES   = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'kills', 'killStreaks',
                'longestKill', 'matchDuration', 'revives', 'rideDistance', 
                'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired']
TARGET     = "winPlacePerc"
ID_COLUMNS = ['matchId', 'groupId', 'maxPlace']         # не фичи модели, нужны для сплита и снапа
EXCLUDED   = ['killPlace', 'rankPoints', 'numGroups',
               'killPoints', 'winPoints']               # исключенные признаки

MATRIX_COLUMNS = ID_COLUMNS + FEATURES + [TARGET]       # полная схема матрицы
