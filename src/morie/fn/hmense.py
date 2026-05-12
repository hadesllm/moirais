# morie.fn -- function file (hadesllm/morie)
"""Ensemble evaluation: average or vote predictions across several models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ensemble_eval"]


def geron_ensemble_eval(models, X):
    """
    Ensemble evaluation: average or vote predictions across several models

    Formula: y_hat = (1/M) sum_{m=1..M} y_hat_m(x)

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
    Géron Ch 2
    """
    models = np.atleast_1d(np.asarray(models, dtype=float))
    n = len(models)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Ensemble evaluation: average or vote predictions across several models"})
    estimate = np.median(models)
    se = 1.2533 * np.std(models, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Ensemble evaluation: average or vote predictions across several models"})


def cheatsheet():
    return "hmense: Ensemble evaluation: average or vote predictions across several models"
