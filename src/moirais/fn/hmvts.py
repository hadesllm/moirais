# moirais.fn — function file (hadesllm/moirais)
"""Soft voting classifier: average predicted class probabilities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_voting_soft"]


def geron_voting_soft(models, X):
    """
    Soft voting classifier: average predicted class probabilities

    Formula: y_hat = argmax_k (1/M) sum_m p_m(k|x)

    Parameters
    ----------
    models : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Géron Ch 6
    """
    models = np.atleast_1d(np.asarray(models, dtype=float))
    n = len(models)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Soft voting classifier: average predicted class probabilities"})
    estimate = np.median(models)
    se = 1.2533 * np.std(models, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Soft voting classifier: average predicted class probabilities"})


def cheatsheet():
    return "hmvts: Soft voting classifier: average predicted class probabilities"
