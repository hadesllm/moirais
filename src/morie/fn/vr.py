"""Variance ratio (F-test for equality of variances)."""

from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from ._containers import ESRes
from ._helpers import _arr


def variance_ratio(x: Union[np.ndarray, pd.Series, list], y: Union[np.ndarray, pd.Series, list], confidence: float = 0.95, cdf=None) -> ESRes:
    """Variance ratio (F-test for equality of variances).

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    x, y = _arr(x), _arr(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    f_val = v1 / v2 if v2 > 0 else np.inf
    df1, df2 = len(x) - 1, len(y) - 1
    alpha = (1 - confidence) / 2
    ci_lo = f_val / stats.f.ppf(1 - alpha, df1, df2)
    ci_hi = f_val / stats.f.ppf(alpha, df1, df2)
    p_val = 2 * min(stats.f.cdf(f_val, df1, df2), stats.f.sf(f_val, df1, df2))
    return ESRes(
        measure="Variance ratio (F)",
        estimate=float(f_val),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=len(x) + len(y),
        extra={"p_value": float(p_val), "df1": df1, "df2": df2},
    )


vr = variance_ratio


def cheatsheet() -> str:
    return "variance_ratio({}) -> Variance ratio (F-test for equality of variances)."
