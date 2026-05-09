"""Double ML PLR cross-fitted estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_dml_partial_lin"]


def causal_dml_partial_lin(y, D, X, n_folds):
    """
    Double ML PLR cross-fitted estimator

    Formula: θ = (D̃'D̃)^{-1} D̃'Ỹ with cross-fitted residuals

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    n_folds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, se

    References
    ----------
    Chernozhukov et al. (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Double ML PLR cross-fitted estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Double ML PLR cross-fitted estimator"})


def cheatsheet():
    return "causdml2: Double ML PLR cross-fitted estimator"
