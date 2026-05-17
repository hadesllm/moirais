"""Kolmogorov-Smirnov one-sample supremum test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import TestResult
from ._helpers import _extract_col


def ks_supremum(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    dist: str = "norm",
    alternative: str = "two-sided",
) -> TestResult:
    """Kolmogorov-Smirnov one-sample supremum test.

    Tests whether the data come from a specified continuous distribution by
    computing the supremum of the absolute difference between the empirical
    and theoretical CDFs.

    Parameters
    ----------
    data : DataFrame or array
        Input data.
    col : str
        Column name if *data* is a DataFrame.
    dist : str
        Scipy distribution name (e.g. ``'norm'``, ``'expon'``, ``'uniform'``).
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.

    Returns
    -------
    TestResult
    """
    x = _extract_col(data, col)
    if len(x) < 5:
        raise ValueError("Need at least 5 observations for KS test")
    try:
        dist_obj = getattr(stats, dist)
    except AttributeError:
        raise ValueError(f"Unknown distribution: {dist}")
    params = dist_obj.fit(x)
    result = stats.kstest(x, dist, args=params, alternative=alternative)
    return TestResult(
        test_name=f"KS supremum test ({dist})",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        method=f"Kolmogorov-Smirnov vs {dist}",
        n=len(x),
        extra={"dist": dist, "fit_params": params, "alternative": alternative},
    )


kssup = ks_supremum


def cheatsheet() -> str:
    return 'ks_supremum({}) -> Supremum test / KS statistic.'
