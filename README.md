# PUBG Placement Predictor — модель + мониторинг дрифта

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/pkg-uv-DE5FE9)
![LightGBM](https://img.shields.io/badge/model-LightGBM-9ACD32)
![Optuna](https://img.shields.io/badge/tuning-Optuna-1E90FF)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow&logoColor=white)

Предсказание финального места игрока в PUBG (`winPlacePerc`) на статическом
Kaggle-датасете **плюс эксперимент по дрифту и деградации модели** на живых матчах из
PUBG API.

Идея: обучаем и «замораживаем» модель на Kaggle-данных 2018 года, затем проверяем её на собранных через API. Игра за годы изменилась, поэтому
дрифт заложен в дизайн: измеряем его (PSI/KS), показываем деградацию качества (MAE) и
переобучаем модель на live, чтобы оценить, сколько потерь отыгрывается.

## Ключевые результаты

Замороженная модель на живых данных деградирует более чем вдвое. Переобучение на live отыгрывает часть ошибки.

| Модель | Данные | MAE |
|---|---|---|
| LightGBM + Optuna (frozen) | Kaggle test | **0.0575** |
| frozen | live | **0.1251** |
| переобучена на live | live test | **0.1099** |

- **Деградация:** `0.0575 -> 0.1251` = **+117%** при переносе на сегодняшний ranked.
- **Выигрыш переобучения:** `0.1251 -> 0.1099`, отыграно `0.0152` (−12% относительно frozen на live).

**Дрифт данных** (reference = Kaggle, current = live, окно 300 ranked squad-fpp матчей):
6 из 11 сырых фич — в сильном дрифте (PSI > 0.25).

| Фича | PSI | Уровень |
|---|---|---|
| matchDuration | 2.92 | strong |
| weaponsAcquired | 1.78 | strong |
| rideDistance | 1.24 | strong |
| boosts | 1.01 | strong |
| walkDistance | 0.57 | strong |
| heals | 0.36 | strong |
| longestKill | 0.11 | moderate |

Графики (наложение распределений обеих выборок, PSI-бары, сдвиг таргета, сравнение MAE) —
в [`notebooks/02_monitoring.ipynb`](notebooks/02_monitoring.ipynb).


## Стек

Python 3.12 · pandas / numpy · LightGBM · Optuna ·
scikit-learn (GroupKFold split) · MLflow (sqlite backend) · httpx (PUBG API-клиент) ·
matplotlib / seaborn.

## Структура

```
src/pubg_ds/
  config.py            # пути, .env, params.yaml
  features/
    schema.py          # единая схема фич — источник правды для reference и live
    derive.py          # производные match/group-relative фичи (общий код)
    reference.py       # reference.parquet из Kaggle train_V2.csv
    live.py            # live.parquet из json PUBG API (parity со схемой)
    split.py           # сплит по matchId (GroupShuffleSplit / GroupKFold)
  models/
    tune.py            # подбор гиперпараметров Optuna (GroupKFold по matchId)
    train.py           # обучение frozen-модели + MLflow
    baseline.py        # нетюненый LightGBM — baseline
    predict.py         # инференс + снап предсказаний к сетке maxPlace
    retrain.py         # переобучение на live + сравнение MAE
  collect/
    client.py          # httpx-клиент
    matches.py         # сбор ranked-матчей
  monitor/
    drift.py           # PSI / KS по фичам
    performance.py     # live-MAE frozen-модели vs Kaggle-test
notebooks/
  01_eda.ipynb         # разведка данных, схема фич, утечки, baseline
  02_monitoring.ipynb  # графики дрифта и деградации
params.yaml            # все гиперпараметры и пороги
Makefile               # оркестрация пайплайна
```

## Установка

Нужен [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync                       # создаёт .venv по pyproject.toml
cp .env.example .env          # затем впиши PUBG_API_KEY (developer.pubg.com → Create App)
```

**Данные Kaggle** 

[PUBG Finish Placement Prediction](https://www.kaggle.com/c/pubg-finish-placement-prediction/data)
`train_V2.csv` в `data/raw/`.

`PUBG_API_KEY` нужен только для live-ветки (`collect`/`live`); reference/train работают без него.

## Запуск

Пайплайн собран в `Makefile`. Порядок:

```bash
# --- Kaggle-ветка: reference-матрица и frozen-модель ---
make reference     # data/processed/reference.parquet из train_V2.csv
make tune          # Optuna-поиск -> models/best_params.json (можно пропустить: параметры уже в репо)
make train         # frozen-модель + метрики + ран в MLflow

# --- live-ветка: сбор и фичи с PUBG API ---
make collect       # окно ranked-матчей -> data/raw/live/*.json  (нужен PUBG_API_KEY)
make live          # data/processed/live.parquet (parity со схемой reference)

# --- мониторинг и переобучение ---
make monitor       # таблица PSI/KS + live-MAE frozen-модели в консоль
make retrain       # переобучение на live + сравнение MAE → reports/mae_comparison.json

make ui            # MLflow UI на http://localhost:5000 
```

Визуализацию дрифта в `notebooks/02_monitoring.ipynb`.

MLflow-бэкенд задаётся через `MLFLOW_TRACKING_URI` (по умолчанию `sqlite:///mlflow.db`);


## Инженерные решения


- **Защита от утечки.** `killPlace` исключён из фич (на игроках с 0 килов corr с таргетом
  ≈ −0.94 — тай-брейк кодирует порядок выбывания). `maxPlace` оставлен как метаданные для
  снапа предсказаний к сетке таргета.
- **Сплит по `matchId` (GroupKFold).** Строки одного матча взаимозависимы, случайный сплит
  завышает метрику; Optuna-objective считается на групповых фолдах, а не на одном сплите.
- **Снап к сетке.** Предсказания округляются к достижимым значениям `k/(maxPlace−1)` —
  единый постпроцессинг в `predict.py`, одинаковый для train, live и retrain.
- **Сбор из API.** Rate limit 10 req/min только на служебных endpoint'ах (`/players`,
  leaderboards); `/matches` вне лимита. Клиент делает retry на 429 (по `Retry-After`) и на
  транзиентные сетевые ошибки, кэширует ответы на диск — прогон возобновляем.
- **Мониторинг.** PSI с бинами по квантилям reference и eps-сглаживанием нулевых долей;
  KS-статистика; пороги дрифта (0.1 / 0.25) вынесены в `params.yaml`.

---

Датасет: [PUBG Finish Placement Prediction](https://www.kaggle.com/c/pubg-finish-placement-prediction) (Kaggle).

API: [PUBG Developer Portal](https://developer.pubg.com/).