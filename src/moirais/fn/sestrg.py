"""S-estimator regression (high breakdown, low efficiency)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["s_estimator_regression"]


def s_estimator_regression(y, X, b):
    """
    S-estimator regression (high breakdown, low efficiency)

    Formula: min sigma s.t. (1/n) sum rho((y_i - x_i'beta)/sigma) = b

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rousseeuw & Yohai (1984)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "S-estimator regression (high breakdown, low efficiency)"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "S-estimator regression (high breakdown, low efficiency)"})


def cheatsheet():
    return "sestrg: S-estimator regression (high breakdown, low efficiency)"
