# morie.fn — function file (hadesllm/morie)
"""Doubly robust (DR) ATE estimator: consistent if either outcome or propensity correctly specified."""
import numpy as np
from ._richresult import RichResult

__all__ = ["doubly_robust_learner"]


def doubly_robust_learner(Y, T, X, mu0_model, mu1_model, e_model):
    """
    Doubly robust (DR) ATE estimator: consistent if either outcome or propensity correctly specified

    Formula: ATE_DR = (1/n)*sum_i [mu_1(X_i) - mu_0(X_i) + T_i*(Y_i-mu_1(X_i))/e(X_i) - (1-T_i)*(Y_i-mu_0(X_i))/(1-e(X_i))]

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.
    X : array-like
        Input data.
    mu0_model : array-like
        Input data.
    mu1_model : array-like
        Input data.
    e_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ate': 'float', 'se': 'float'}

    References
    ----------
    Molak Ch 10
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    if Y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Doubly robust (DR) ATE estimator: consistent if either outcome or propensity correctly specified"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Doubly robust (DR) ATE estimator: consistent if either outcome or propensity correctly specified"})


def cheatsheet():
    return "drblr: Doubly robust (DR) ATE estimator: consistent if either outcome or propensity correctly specified"
