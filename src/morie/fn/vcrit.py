"""Criterion validity: correlation with external criterion."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp
from ._richresult import RichResult


def validity_criterion(
    scores: np.ndarray,
    criterion: np.ndarray,
    *,
    ci: float = 0.95,
) -> dict:
    """Criterion (concurrent) validity via Pearson correlation.

    Correlates scale scores with an external criterion measure.

    Parameters
    ----------
    scores : array-like
        Scale or subscale scores.
    criterion : array-like
        External criterion variable.
    ci : float
        Confidence level for the interval (default 0.95).

    Returns
    -------
    dict
        Keys: ``r``, ``p_value``, ``ci_lower``, ``ci_upper``, ``n``.

    References
    ----------
    American Educational Research Association. (2014). *Standards for
    Educational and Psychological Testing*. AERA.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    c = np.asarray(criterion, dtype=np.float64).ravel()
    mask = np.isfinite(s) & np.isfinite(c)
    s, c = s[mask], c[mask]
    n = len(s)
    if n < 3:
        return RichResult(payload={"r": np.nan, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": n})

    r, p = sp.pearsonr(s, c)
    # Fisher z transformation for CI
    z = np.arctanh(r)
    se_z = 1.0 / np.sqrt(n - 3)
    alpha = 1 - ci
    z_crit = sp.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se_z)
    ci_hi = np.tanh(z + z_crit * se_z)

    return {
        "r": float(r),
        "p_value": float(p),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n": n,
    }


def cheatsheet() -> str:
    return "validity_criterion({}) -> Criterion validity: correlation with external criterion."
