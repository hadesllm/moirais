# morie.fn — function file (hadesllm/morie)
"""Summary of all effect sizes for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from ._richresult import RichResult


def otis_effect_summary(
    df: pd.DataFrame,
    *,
    outcome: str = "Y",
    treatment: str = "D",
) -> dict:
    """Compute summary of standard effect sizes.

    Calculates Cohen's d, point-biserial r, odds ratio, and risk
    ratio (for binary outcomes) between treatment groups.

    Parameters
    ----------
    df : DataFrame
        Data with outcome and treatment columns.
    outcome, treatment : str
        Column names.

    Returns
    -------
    dict
        Keys: cohens_d, point_biserial_r, odds_ratio, risk_ratio, n.
        OR and RR are NaN for continuous outcomes.
    """
    data = df[[outcome, treatment]].dropna()
    t1 = data.loc[data[treatment] == 1, outcome].values.astype(np.float64)
    t0 = data.loc[data[treatment] == 0, outcome].values.astype(np.float64)
    n = len(data)

    if len(t1) < 2 or len(t0) < 2:
        return RichResult(payload={"cohens_d": np.nan, "point_biserial_r": np.nan, "odds_ratio": np.nan, "risk_ratio": np.nan, "n": n})

    # Cohen's d (pooled SD)
    s_pooled = np.sqrt(((len(t1) - 1) * t1.var(ddof=1) + (len(t0) - 1) * t0.var(ddof=1)) / (len(t1) + len(t0) - 2))
    d = float((t1.mean() - t0.mean()) / s_pooled) if s_pooled > 0 else 0.0

    # Point-biserial r
    r, _ = stats.pointbiserialr(data[treatment].values, data[outcome].values)

    # OR and RR (only for binary outcomes)
    unique_vals = set(data[outcome].unique())
    if unique_vals <= {0, 1}:
        a = np.sum((data[treatment] == 1) & (data[outcome] == 1))
        b = np.sum((data[treatment] == 1) & (data[outcome] == 0))
        c = np.sum((data[treatment] == 0) & (data[outcome] == 1))
        dd = np.sum((data[treatment] == 0) & (data[outcome] == 0))
        odds_ratio = float((a * dd) / (b * c)) if (b * c) > 0 else np.nan
        p1 = a / (a + b) if (a + b) > 0 else 0
        p0 = c / (c + dd) if (c + dd) > 0 else 0
        risk_ratio = float(p1 / p0) if p0 > 0 else np.nan
    else:
        odds_ratio = np.nan
        risk_ratio = np.nan

    return {
        "cohens_d": round(d, 4),
        "point_biserial_r": round(float(r), 4),
        "odds_ratio": round(odds_ratio, 4) if np.isfinite(odds_ratio) else np.nan,
        "risk_ratio": round(risk_ratio, 4) if np.isfinite(risk_ratio) else np.nan,
        "n": n,
    }


def cheatsheet() -> str:
    return "otis_effect_summary({}) -> Summary of all effect sizes for OTIS correctional data."
