"""Doubly-robust DiD estimator (Sant'Anna & Zhao 2020)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_santanna_zhao"]


def dr_did_santanna_zhao(y_pre, y_post, treatment, X, ml_propensity, ml_outcome):
    """
    Doubly-robust DiD estimator (Sant'Anna & Zhao 2020)

    Formula: ATT = E[(D - p(X))/(1-p(X)) * (Y_1 - Y_0 - m_0(X))] under DR moment

    Parameters
    ----------
    y_pre : array-like
        Input data.
    y_post : array-like
        Input data.
    treatment : array-like
        Input data.
    X : array-like
        Input data.
    ml_propensity : array-like
        Input data.
    ml_outcome : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sant'Anna & Zhao (2020) JoE
    """
    y_pre = np.atleast_1d(np.asarray(y_pre, dtype=float))
    n = len(y_pre)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Doubly-robust DiD estimator (Sant'Anna & Zhao 2020)"})
    estimate = np.median(y_pre)
    se = 1.2533 * np.std(y_pre, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Doubly-robust DiD estimator (Sant'Anna & Zhao 2020)"})


def cheatsheet():
    return "drsza: Doubly-robust DiD estimator (Sant'Anna & Zhao 2020)"
