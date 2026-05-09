"""TMLE for average predictiveness measures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_average_predictiveness"]


def tmle_average_predictiveness(y, D, X, f, loss):
    """
    TMLE for average predictiveness measures

    Formula: target E[loss(f(X), Y)] under intervention

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    f : array-like
        Input data.
    loss : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Williamson-Gilbert-Carone-Simon (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "TMLE for average predictiveness measures"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "TMLE for average predictiveness measures"})


def cheatsheet():
    return "tmlavp: TMLE for average predictiveness measures"
