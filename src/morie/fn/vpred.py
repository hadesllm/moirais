"""Predictive validity: AUC or R-squared for outcome prediction."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp
from ._richresult import RichResult


def validity_predictive(
    scores: np.ndarray,
    outcome: np.ndarray,
    *,
    method: str = "logistic",
) -> dict:
    """Predictive validity of scale scores against an outcome.

    For binary outcomes (``method='logistic'``), reports AUC via the
    Mann-Whitney U statistic.  For continuous outcomes
    (``method='linear'``), reports R-squared.

    Parameters
    ----------
    scores : array-like
        Scale scores (predictor).
    outcome : array-like
        Outcome variable (binary 0/1 or continuous).
    method : str
        ``'logistic'`` (default) for binary AUC, ``'linear'`` for R-squared.

    Returns
    -------
    dict
        Keys depend on method: ``auc`` or ``r_squared``, plus ``n``, ``method``.

    References
    ----------
    Hosmer, D. W., & Lemeshow, S. (2000). *Applied Logistic Regression*.
    Wiley.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(outcome, dtype=np.float64).ravel()
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    n = len(s)

    if method == "logistic":
        # AUC via Mann-Whitney U
        pos = s[y == 1]
        neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return RichResult(payload={"auc": np.nan, "n": n, "method": method})
        u_stat, _ = sp.mannwhitneyu(pos, neg, alternative="two-sided")
        auc = u_stat / (len(pos) * len(neg))
        return RichResult(payload={"auc": float(auc), "n": n, "n_pos": len(pos), "n_neg": len(neg), "method": method})
    else:
        # Linear: R-squared
        if n < 3:
            return RichResult(payload={"r_squared": np.nan, "n": n, "method": method})
        r, p = sp.pearsonr(s, y)
        return RichResult(payload={"r_squared": float(r**2), "r": float(r), "p_value": float(p), "n": n, "method": method})


def cheatsheet() -> str:
    return "validity_predictive({}) -> Predictive validity: AUC or R-squared for outcome prediction"
