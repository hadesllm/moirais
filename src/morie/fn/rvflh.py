# morie.fn -- function file (hadesllm/morie)
"""The man who moves a mountain begins by carrying away small stones. -- Confucius"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import TestResult
from ._helpers import _validate_df


def causal_reversal_test(
    data: pd.DataFrame,
    *,
    x: str = "x",
    y: str = "y",
    n_perm: int = 999,
    seed: int | None = None,
) -> TestResult:
    """Test whether the causal direction X->Y is more plausible than Y->X.

    Uses the asymmetry of regression residuals: if X truly causes Y, then
    regressing Y on X should yield more Gaussian residuals than regressing
    X on Y (Shimizu et al., 2006).  Compares Shapiro-Wilk statistics of
    forward and reverse residuals.

    Parameters
    ----------
    data : DataFrame
        Must contain columns *x* and *y*.
    x, y : str
        Column names for putative cause and effect.
    n_perm : int
        Permutations for significance testing.
    seed : int or None
        Random seed.

    Returns
    -------
    TestResult
        Statistic > 0 suggests X->Y; < 0 suggests Y->X.
    """
    _validate_df(data, x, y)
    df = data[[x, y]].dropna()
    xv = df[x].to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float)
    n = len(xv)
    if n < 10:
        raise ValueError("Need at least 10 observations")
    b_xy = np.polyfit(xv, yv, 1)
    r_xy = yv - np.polyval(b_xy, xv)
    b_yx = np.polyfit(yv, xv, 1)
    r_yx = xv - np.polyval(b_yx, yv)
    cap = min(n, 5000)
    sw_xy = stats.shapiro(r_xy[:cap]).statistic
    sw_yx = stats.shapiro(r_yx[:cap]).statistic
    obs_stat = float(sw_xy - sw_yx)
    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        yp = yv[perm]
        b1 = np.polyfit(xv, yp, 1)
        r1 = yp - np.polyval(b1, xv)
        b2 = np.polyfit(yp, xv, 1)
        r2 = xv - np.polyval(b2, yp)
        null_stats[i] = stats.shapiro(r1[:cap]).statistic - stats.shapiro(r2[:cap]).statistic
    p_value = float((np.abs(null_stats) >= abs(obs_stat)).sum() + 1) / (n_perm + 1)
    direction = "X->Y" if obs_stat > 0 else "Y->X" if obs_stat < 0 else "indeterminate"
    return TestResult(
        test_name="Causal reversal test",
        statistic=obs_stat,
        p_value=p_value,
        method="Residual normality asymmetry",
        n=n,
        extra={
            "sw_forward": float(sw_xy),
            "sw_reverse": float(sw_yx),
            "direction": direction,
            "n_perm": n_perm,
        },
    )


rvflh = causal_reversal_test


def cheatsheet() -> str:
    return "causal_reversal_test({}) -> Causal reversal test. 'It was me, Barry. It was me all along"
