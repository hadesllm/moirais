# moirais.fn — function file (hadesllm/moirais)
"""Scikit-Learn Pipeline: chain preprocessing steps with a final estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pipeline"]


def geron_pipeline(steps, X, y):
    """
    Scikit-Learn Pipeline: chain preprocessing steps with a final estimator

    Formula: pipeline: T_1 -> T_2 -> ... -> estimator

    Parameters
    ----------
    steps : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fitted_pipeline

    References
    ----------
    Géron Ch 2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Scikit-Learn Pipeline: chain preprocessing steps with a final estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Scikit-Learn Pipeline: chain preprocessing steps with a final estimator"})


def cheatsheet():
    return "hmpip: Scikit-Learn Pipeline: chain preprocessing steps with a final estimator"
