# morie.fn -- function file (hadesllm/morie)
"""Overall recidivism rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS
from ._richresult import RichResult


def recidivism_rate(
    df: pd.DataFrame,
    *,
    outcome: str = DEFAULT_COLS["outcome"],
    threshold: float = 0.0,
) -> dict:
    """Overall recidivism rate (proportion above threshold).

    Parameters
    ----------
    df : DataFrame
        Dataset with outcome column.
    outcome : str
        Column with recidivism outcome (continuous or binary).
    threshold : float
        Value above which a record counts as recidivism.

    Returns
    -------
    dict
        rate, n_recid, n_total, ci_lower, ci_upper (Wald 95% CI).
    """
    vals = df[outcome].dropna()
    n_total = len(vals)
    if n_total == 0:
        return RichResult(payload={"rate": np.nan, "n_recid": 0, "n_total": 0, "ci_lower": np.nan, "ci_upper": np.nan})
    n_recid = int((vals > threshold).sum())
    rate = n_recid / n_total
    se = np.sqrt(rate * (1 - rate) / n_total) if n_total > 0 else 0.0
    ci_lower = max(0.0, rate - 1.96 * se)
    ci_upper = min(1.0, rate + 1.96 * se)
    return {
        "rate": rate,
        "n_recid": n_recid,
        "n_total": n_total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


rcdsm = recidivism_rate


def cheatsheet() -> str:
    return "recidivism_rate({}) -> Overall recidivism rate."
