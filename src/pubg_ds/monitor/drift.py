"""Data drift: PSI и KS по каждой фиче (reference = Kaggle, current = live).

PSI = Σ (aᵢ − eᵢ)·ln(aᵢ/eᵢ), бины — по квантилям reference, нулевые доли
сглаживаются eps. Пороги: < 0.1 стабильно, 0.1–0.25 умеренный сдвиг,
>= 0.25 сильный дрифт.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from pubg_ds import config
from pubg_ds.features import schema


# Population Stability Index; бины фиксируются по квантилям reference
def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    # Квантильные границы; крайние — в бесконечность, чтобы покрыть хвосты current
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0            # константная фича — бинов не построить
    edges[0], edges[-1] = -np.inf, np.inf

    # Доли по бинам; нулевые сглаживаем eps, иначе ln взрывается
    e = np.histogram(reference, edges)[0] / len(reference)
    a = np.histogram(current, edges)[0] / len(current)
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


# KS-статистика: sup |F_ref(x) − F_live(x)|
def ks(reference: np.ndarray, current: np.ndarray) -> float:
    return float(ks_2samp(reference, current).statistic)


# Вербальный вердикт по порогам PSI
def psi_level(p: float, psi_moderate: float, psi_strong: float) -> str:
    if p >= psi_strong:
        return "strong"
    if p >= psi_moderate:
        return "moderate"
    return "stable"


# PSI/KS по всем сырым фичам схемы + вердикт по порогам
def drift_table(
    ref: pd.DataFrame, cur: pd.DataFrame, bins: int, eps: float,
    psi_moderate: float, psi_strong: float,
) -> pd.DataFrame:
    rows = []
    for feat in schema.FEATURES:
        r = ref[feat].to_numpy(dtype=float)
        c = cur[feat].to_numpy(dtype=float)
        p = psi(r, c, bins, eps)
        rows.append({
            "feature": feat,
            "psi":     p,
            "ks":      ks(r, c),
            "level":   psi_level(p, psi_moderate, psi_strong),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# Удобный вход для ноутбука/CLI: грузит parquet-ы и params, возвращает таблицу
def compute() -> pd.DataFrame:
    p   = config.load_params()["monitor"]
    ref = pd.read_parquet(config.DATA_PROCESSED / "reference.parquet")
    cur = pd.read_parquet(config.DATA_PROCESSED / "live.parquet")
    return drift_table(ref, cur, p["psi_bins"], p["psi_eps"], p["psi_moderate"], p["psi_strong"])


def main() -> None:
    table = compute()
    print(table.to_string(index=False))
    n = table["level"].value_counts()
    print(f"\nstrong={n.get('strong', 0)}  moderate={n.get('moderate', 0)}  stable={n.get('stable', 0)}")


if __name__ == "__main__":
    main()
