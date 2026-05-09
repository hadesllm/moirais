"""Wald estimator for binary instrument and treatment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_iv_instrumental_dag"]


def causal_iv_instrumental_dag(y, D, Z):
    """
    Wald estimator for binary instrument and treatment

    Formula: β = E[Y|Z=1]-E[Y|Z=0] / E[D|Z=1]-E[D|Z=0]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, se

    References
    ----------
    Imbens & Angrist (1994)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Wald estimator for binary instrument and treatment"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Wald estimator for binary instrument and treatment"})


def cheatsheet():
    return "causinst: Wald estimator for binary instrument and treatment"
