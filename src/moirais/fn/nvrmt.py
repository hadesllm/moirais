# moirais.fn — function file (hadesllm/moirais)
"""Normal variance ratio test (F-test for equal variances)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def normal_var_ratio_test(x, y, cdf=None, *, alternative: str = "two-sided") -> TestResult:
    """F-test for equality of two normal variances.

    Parameters
    ----------
    x, y : array-like
        Two independent normal samples.
    alternative : str
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    TestResult
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("Both samples need at least 2 observations.")

    v1 = np.var(a, ddof=1)
    v2 = np.var(b, ddof=1)
    f_stat = v1 / v2 if v2 > 1e-12 else np.inf
    df1, df2 = n1 - 1, n2 - 1

    if alternative == "two-sided":
        pval = 2.0 * min(
            sp_stats.f.cdf(f_stat, df1, df2),
            sp_stats.f.sf(f_stat, df1, df2),
        )
    elif alternative == "greater":
        pval = sp_stats.f.sf(f_stat, df1, df2)
    else:
        pval = sp_stats.f.cdf(f_stat, df1, df2)

    return TestResult(
        test_name="Variance ratio F-test",
        statistic=float(f_stat),
        p_value=float(pval),
        df=float(df1),
        n=n1 + n2,
        method=f"alternative={alternative}",
        extra={"df1": df1, "df2": df2, "var1": float(v1), "var2": float(v2)},
    )


nvrmt = normal_var_ratio_test


def cheatsheet() -> str:
    return "normal_var_ratio_test(x, y) -> F-test for equal variances."
