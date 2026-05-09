"""Test-retest reliability: ICC and Pearson r."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp
from ._richresult import RichResult


def validity_test_retest(
    scores1: np.ndarray,
    scores2: np.ndarray,
    *,
    ci: float = 0.95,
) -> dict:
    """Test-retest reliability via ICC(3,1) and Pearson r.

    ICC(3,1) — two-way mixed, consistency, single measures — is the
    standard for test-retest (Shrout & Fleiss, 1979).

    Parameters
    ----------
    scores1 : array-like
        Scores from time 1.
    scores2 : array-like
        Scores from time 2 (same subjects).
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    dict
        Keys: ``icc``, ``icc_ci_lower``, ``icc_ci_upper``, ``pearson_r``,
        ``pearson_p``, ``mean_diff``, ``sd_diff``, ``n``.

    References
    ----------
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations:
    uses in assessing rater reliability. *Psychological Bulletin*,
    86(2), 420--428.
    """
    s1 = np.asarray(scores1, dtype=np.float64).ravel()
    s2 = np.asarray(scores2, dtype=np.float64).ravel()
    mask = np.isfinite(s1) & np.isfinite(s2)
    s1, s2 = s1[mask], s2[mask]
    n = len(s1)

    if n < 3:
        return RichResult(payload={"icc": np.nan, "pearson_r": np.nan, "n": n})

    # Pearson r
    r, p_r = sp.pearsonr(s1, s2)

    # ICC(3,1) — two-way mixed, consistency
    # Using two-way ANOVA decomposition
    k = 2  # number of measurements
    grand_mean = np.mean(np.concatenate([s1, s2]))
    row_means = (s1 + s2) / 2.0
    col_means = np.array([np.mean(s1), np.mean(s2)])

    ss_row = k * np.sum((row_means - grand_mean) ** 2)
    ss_col = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((s1 - grand_mean) ** 2) + np.sum((s2 - grand_mean) ** 2)
    ss_error = ss_total - ss_row - ss_col

    ms_row = ss_row / (n - 1) if n > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0

    icc = (ms_row - ms_error) / (ms_row + (k - 1) * ms_error) if (ms_row + (k - 1) * ms_error) > 0 else 0.0

    # CI for ICC via F distribution (Shrout & Fleiss)
    alpha = 1 - ci
    f_val = ms_row / ms_error if ms_error > 0 else np.inf
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    f_lo = f_val / sp.f.ppf(1 - alpha / 2, df1, df2)
    f_hi = f_val / sp.f.ppf(alpha / 2, df1, df2)
    icc_lo = (f_lo - 1) / (f_lo + k - 1)
    icc_hi = (f_hi - 1) / (f_hi + k - 1)

    diff = s1 - s2
    return {
        "icc": float(icc),
        "icc_ci_lower": float(icc_lo),
        "icc_ci_upper": float(icc_hi),
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "mean_diff": float(np.mean(diff)),
        "sd_diff": float(np.std(diff, ddof=1)),
        "n": n,
    }


def cheatsheet() -> str:
    return "validity_test_retest({}) -> Test-retest reliability: ICC and Pearson r."
