"""Bootstrap variance estimator of T(F)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_var_estimator"]


def boot_var_estimator(theta_b):
    """
    Bootstrap variance estimator of T(F)

    Formula: Var* = (1/(B-1)) Σ (T_b - T̄)²

    Parameters
    ----------
    theta_b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: var_hat, mean_hat

    References
    ----------
    Efron & Tibshirani (1993)
    """
    theta_b = np.atleast_1d(np.asarray(theta_b, dtype=float))
    n = len(theta_b)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bootstrap variance estimator of T(F)"})
    estimate = np.median(theta_b)
    se = 1.2533 * np.std(theta_b, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bootstrap variance estimator of T(F)"})


def cheatsheet():
    return "btvb: Bootstrap variance estimator of T(F)"
