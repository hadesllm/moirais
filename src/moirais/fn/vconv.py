"""Convergent validity evidence."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp

from moirais.fn._containers import ESRes


def convergent_validity(
    test_scores: np.ndarray,
    criterion_scores: np.ndarray,
    *,
    ci: float = 0.95,
) -> ESRes:
    """Convergent validity: correlation between test and criterion.

    Parameters
    ----------
    test_scores : ndarray
        Scores on the test being validated (n,).
    criterion_scores : ndarray
        Scores on the convergent criterion measure (n,).
    ci : float
        Confidence level (default 0.95).

    Returns
    -------
    ESRes
        measure="convergent_validity".

    References
    ----------
    Campbell, D. T. & Fiske, D. W. (1959). Convergent and discriminant
    validation by the multitrait-multimethod matrix. Psychological
    Bulletin, 56(2), 81-105.
    """
    x = np.asarray(test_scores, dtype=np.float64).ravel()
    y = np.asarray(criterion_scores, dtype=np.float64).ravel()
    n = len(x)
    if n != len(y):
        raise ValueError("test_scores and criterion_scores must have same length.")

    r, p_val = sp.pearsonr(x, y)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z = np.arctanh(r)
    z_crit = sp.norm.ppf(1 - (1 - ci) / 2)
    ci_lo = float(np.tanh(z - z_crit * se))
    ci_hi = float(np.tanh(z + z_crit * se))

    return ESRes(
        measure="convergent_validity",
        estimate=float(r),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=float(se),
        n=n,
        extra={"p_value": float(p_val), "r_squared": float(r**2)},
    )


convergent = convergent_validity


def cheatsheet() -> str:
    return "convergent_validity({}) -> Convergent validity evidence."
