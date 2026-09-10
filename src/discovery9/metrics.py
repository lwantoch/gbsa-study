"""BEDROC + rank-fusion helpers.

BEDROC (Boltzmann-Enhanced Discrimination of ROC), Truchon & Bayly 2007.
Higher scores rank first. Returns a scalar in [0, 1] — higher is better.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def bedroc(scores: Iterable[float], labels: Iterable[int], alpha: float = 20.0) -> float:
    """BEDROC α (default α = 20 — used as the confirmatory primary in this study).

    Convention: **higher scores rank first**. If your scorer is "lower = better"
    (e.g. GBSA ΔG, docking score) negate it first: ``bedroc(-scores, labels)``.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[mask], labels[mask]
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    N = len(labels)
    Ra = n_pos / N
    ranks = np.where(labels == 1)[0] + 1
    num = np.sum(np.exp(-alpha * ranks / N))
    denom = Ra * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1)
    Rf = num / denom if denom > 0 else float("nan")
    factor = Ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * Ra))
    return float(Rf * factor + 1 / (1 - np.exp(alpha * (1 - Ra))))


def rank_fuse(df: pd.DataFrame, feature_directions: list[tuple[str, int]]) -> np.ndarray:
    """Rank-fusion (mean rank) of several features on a single-target subframe.

    ``feature_directions`` is a list of ``(column_name, sign)`` where sign = +1 means
    higher-is-better and −1 means lower-is-better. Returns per-row fused rank score
    (higher = ranks nearer top). Suitable as the score argument to ``bedroc()``.
    """
    n = len(df)
    rs = np.zeros(n, dtype=float)
    for col, sign in feature_directions:
        vals = sign * df[col].values
        rs += rankdata(vals, method="average")
    return rs / len(feature_directions)


def bedroc_per_target(df: pd.DataFrame, scorer_col: str, label_col: str = "is_active",
                      target_col: str = "target", alpha: float = 20.0,
                      lower_is_better: bool = False) -> pd.Series:
    """Compute per-target BEDROC α of a scorer column. Returns a Series indexed by target."""
    sign = -1 if lower_is_better else 1
    return df.groupby(target_col).apply(
        lambda g: bedroc(sign * g[scorer_col].values, g[label_col].astype(int).values, alpha=alpha),
        include_groups=False,
    )
