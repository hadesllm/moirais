# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""No man ever steps in the same river twice. — Heraclitus"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import TestResult
from ._helpers import _extract_col


def structural_break(data: pd.DataFrame | np.ndarray, cdf=None, *, col: str = "x", min_segment: int = 10) -> TestResult:
    """Chow-type structural break test using maximum F-statistic.

    Scans all candidate breakpoints and returns the location with the largest
    F-statistic comparing the full-sample regression to the two sub-sample
    regressions (linear trend model).

    Parameters
    ----------
    data : DataFrame or array
        Time series data.
    col : str
        Column name if *data* is a DataFrame.
    min_segment : int
        Minimum observations in each segment.

    Returns
    -------
    TestResult
        Statistic is the maximum F, p-value from the F-distribution.
    """
    x = _extract_col(data, col)
    n = len(x)
    if n < 2 * min_segment:
        raise ValueError(f"Need at least {2 * min_segment} observations")
    t_full = np.arange(n, dtype=float)
    X_full = np.column_stack([np.ones(n), t_full])
    beta_full = np.linalg.lstsq(X_full, x, rcond=None)[0]
    rss_full = float(np.sum((x - X_full @ beta_full) ** 2))
    k = 2
    best_f = -1.0
    best_bp = min_segment
    for bp in range(min_segment, n - min_segment + 1):
        x1, x2 = x[:bp], x[bp:]
        t1 = np.arange(bp, dtype=float)
        t2 = np.arange(len(x2), dtype=float)
        X1 = np.column_stack([np.ones(bp), t1])
        X2 = np.column_stack([np.ones(len(x2)), t2])
        b1 = np.linalg.lstsq(X1, x1, rcond=None)[0]
        b2 = np.linalg.lstsq(X2, x2, rcond=None)[0]
        rss1 = float(np.sum((x1 - X1 @ b1) ** 2))
        rss2 = float(np.sum((x2 - X2 @ b2) ** 2))
        rss_ur = rss1 + rss2
        if rss_ur < 1e-30:
            continue
        f_stat = ((rss_full - rss_ur) / k) / (rss_ur / (n - 2 * k))
        if f_stat > best_f:
            best_f = f_stat
            best_bp = bp
    df1, df2 = k, n - 2 * k
    p_value = float(1 - stats.f.cdf(best_f, df1, df2)) if df2 > 0 else 1.0
    return TestResult(
        test_name="Structural break test (max F)",
        statistic=best_f,
        p_value=p_value,
        df=float(df1),
        method="Chow-type maximum F",
        n=n,
        extra={"breakpoint": best_bp, "df1": df1, "df2": df2, "min_segment": min_segment},
    )


bane = structural_break


def cheatsheet() -> str:
    return "No man ever steps in the same river twice. — Heraclitus"
