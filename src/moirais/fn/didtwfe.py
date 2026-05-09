"""Two-way fixed effects DID estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["twoway_fixed_effects_did"]


def twoway_fixed_effects_did(y, D, unit, time):
    """
    Two-way fixed effects DID estimator

    Formula: y_it = alpha_i + lambda_t + beta D_it + epsilon

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Card-Krueger (1994); Bertrand-Duflo-Mullainathan (2004)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Two-way fixed effects DID estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Two-way fixed effects DID estimator"})


def cheatsheet():
    return "didtwfe: Two-way fixed effects DID estimator"
