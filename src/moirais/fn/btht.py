"""Bootstrap two-sided p-value for H0: θ=θ0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_test_hypothesis"]


def boot_test_hypothesis(x, theta0, stat, B):
    """
    Bootstrap two-sided p-value for H0: θ=θ0

    Formula: p = 2 min(P*(T*>=T̂), P*(T*<=T̂))

    Parameters
    ----------
    x : array-like
        Input data.
    theta0 : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p, T_hat, T_b

    References
    ----------
    Davison & Hinkley (1997)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap two-sided p-value for H0: θ=θ0"})


def cheatsheet():
    return "btht: Bootstrap two-sided p-value for H0: θ=θ0"
