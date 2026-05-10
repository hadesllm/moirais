# moirais.fn — function file (hadesllm/moirais)
"""ATE difference-in-means. 'We suffer more often in imagination than in reality. -- Seneca'"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def ate_diff(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    alpha: float = 0.05,
) -> ESRes:
    """Average treatment effect via simple difference-in-means with Welch CI."""
    _validate_df(data, y, t)
    df = data[[y, t]].dropna()
    y1 = df.loc[df[t] == 1, y].to_numpy(dtype=float)
    y0 = df.loc[df[t] == 0, y].to_numpy(dtype=float)
    if len(y1) < 2 or len(y0) < 2:
        raise ValueError("Need at least 2 obs per group")
    ate = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    df_w = (y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)) ** 2 / (
        (y1.var(ddof=1) / len(y1)) ** 2 / (len(y1) - 1) + (y0.var(ddof=1) / len(y0)) ** 2 / (len(y0) - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df_w)
    return ESRes(
        measure="ATE (diff-in-means)",
        estimate=ate,
        ci_lower=ate - t_crit * se,
        ci_upper=ate + t_crit * se,
        se=se,
        n=len(df),
        extra={"n1": len(y1), "n0": len(y0), "df": df_w},
    )


force = ate_diff


def cheatsheet() -> str:
    return "ate_diff({}) -> ATE difference-in-means. 'The Force is strong with this one."
