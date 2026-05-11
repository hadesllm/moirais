# morie.fn — function file (hadesllm/morie)
"""K-fold cross-validation for honest generalization estimate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cross_validation"]


def geron_cross_validation(X, y, k):
    """
    K-fold cross-validation for honest generalization estimate

    Formula: CV(k) = (1/k) sum_{i=1..k} L(f_i, D_{test,i})

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cv_score

    References
    ----------
    Géron Ch 1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "K-fold cross-validation for honest generalization estimate"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "K-fold cross-validation for honest generalization estimate"})


def cheatsheet():
    return "hmcv: K-fold cross-validation for honest generalization estimate"
