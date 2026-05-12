# morie.fn -- function file (hadesllm/morie)
"""Out of chaos, comes order. -- Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def harrells_c(
    data: pd.DataFrame,
    *,
    time: str = "time",
    event: str = "event",
    risk: str = "risk_score",
) -> DescriptiveResult:
    """Harrell's C concordance index for survival predictions.

    Measures the probability that, for a random pair of observations, the one
    with the higher predicted risk score has the shorter survival time (among
    concordant pairs where the earlier event is uncensored).

    Parameters
    ----------
    data : DataFrame
        Must contain time, event (0/1), and risk score columns.
    time : str
        Survival time column.
    event : str
        Event indicator column (1 = event, 0 = censored).
    risk : str
        Predicted risk score column (higher = worse prognosis).

    Returns
    -------
    DescriptiveResult
        ``value`` = Harrell's C index (0 to 1; 0.5 = random).
    """
    _validate_df(data, time, event, risk)
    df = data[[time, event, risk]].dropna()
    t = df[time].to_numpy(dtype=float)
    e = df[event].to_numpy(dtype=int)
    r = df[risk].to_numpy(dtype=float)
    n = len(t)
    if n < 2:
        raise ValueError("Need at least 2 observations")
    concordant = 0
    discordant = 0
    tied = 0
    for i in range(n):
        if e[i] == 0:
            continue
        for j in range(n):
            if t[j] <= t[i] and i != j:
                continue
            if t[j] > t[i]:
                if r[j] < r[i]:
                    concordant += 1
                elif r[j] > r[i]:
                    discordant += 1
                else:
                    tied += 1
    total = concordant + discordant + tied
    c_index = (concordant + 0.5 * tied) / total if total > 0 else 0.5
    se = np.sqrt(c_index * (1 - c_index) / total) if total > 0 else 0.0
    return DescriptiveResult(
        name="Harrell's C concordance index",
        value=float(c_index),
        extra={
            "concordant": concordant,
            "discordant": discordant,
            "tied": tied,
            "total_pairs": total,
            "se": float(se),
            "n": n,
            "n_events": int(e.sum()),
        },
    )


harqn = harrells_c


def cheatsheet() -> str:
    return "Out of chaos, comes order. -- Friedrich Nietzsche"
