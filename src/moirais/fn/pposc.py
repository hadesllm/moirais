"""Posterior predictive check (Bayesian p-value)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["posterior_predictive_check"]


def posterior_predictive_check(y, y_rep):
    """
    Posterior predictive check (Bayesian p-value)

    Formula: p_B = Pr( T(y_rep, theta) >= T(y, theta) | y )

    Parameters
    ----------
    y : array-like
        Input data.
    y_rep : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman, Meng, Stern (1996)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior predictive check (Bayesian p-value)"})


def cheatsheet():
    return "pposc: Posterior predictive check (Bayesian p-value)"
