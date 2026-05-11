# morie.fn — function file (hadesllm/morie)
"""Test-retest reliability via ICC."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp

from morie.fn._containers import ESRes


def retest_reliability(
    scores_t1: np.ndarray,
    scores_t2: np.ndarray,
    *,
    ci: float = 0.95,
) -> ESRes:
    """Test-retest reliability via intraclass correlation (ICC 2,1).

    Uses two-way random, single measures, absolute agreement.

    Parameters
    ----------
    scores_t1 : ndarray
        Scores at time 1 (n,).
    scores_t2 : ndarray
        Scores at time 2 (n,).
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    ESRes
        measure="test_retest_ICC".

    References
    ----------
    Shrout, P. E. & Fleiss, J. L. (1979). Intraclass correlations:
    Uses in assessing rater reliability. Psychological Bulletin.
    """
    t1 = np.asarray(scores_t1, dtype=np.float64).ravel()
    t2 = np.asarray(scores_t2, dtype=np.float64).ravel()
    n = len(t1)
    if n != len(t2):
        raise ValueError("scores_t1 and scores_t2 must have same length.")

    data = np.column_stack([t1, t2])
    k = 2
    grand_mean = data.mean()
    ss_total = np.sum((data - grand_mean) ** 2)
    row_means = data.mean(axis=1)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    col_means = data.mean(axis=0)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / max(n - 1, 1)
    ms_error = ss_error / max((n - 1) * (k - 1), 1)
    ms_cols = ss_cols / max(k - 1, 1)

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    icc = float(np.clip(icc, -1, 1))

    alpha_half = (1 - ci) / 2
    F_val = ms_rows / max(ms_error, 1e-10)
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    F_lo = F_val / sp.f.ppf(1 - alpha_half, df1, df2)
    F_hi = F_val / sp.f.ppf(alpha_half, df1, df2)
    ci_lo = (F_lo - 1) / (F_lo + k - 1)
    ci_hi = (F_hi - 1) / (F_hi + k - 1)

    return ESRes(
        measure="test_retest_ICC",
        estimate=icc,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n,
        extra={"pearson_r": float(np.corrcoef(t1, t2)[0, 1])},
    )


retest = retest_reliability


def cheatsheet() -> str:
    return "retest_reliability({}) -> Test-retest reliability via ICC."
