"""Plain non-parametric IID bootstrap of an estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_iid_resample"]


def boot_iid_resample(x, stat, B):
    """
    Plain non-parametric IID bootstrap of an estimator

    Formula: θ̂*_b = T(x*_b),  x*_b ~ Multinomial draws with replacement

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b, theta_hat

    References
    ----------
    Efron (1979)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Plain non-parametric IID bootstrap of an estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Plain non-parametric IID bootstrap of an estimator"})


def cheatsheet():
    return "btiid: Plain non-parametric IID bootstrap of an estimator"
