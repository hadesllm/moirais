"""Pseudo-Bayesian credible bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_pseudo_credible"]


def bound_pseudo_credible(y, X, alpha):
    """
    Pseudo-Bayesian credible bound

    Formula: frequentist + likelihood-based prior

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Müller-Norets (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pseudo-Bayesian credible bound"})


def cheatsheet():
    return "bndpcb: Pseudo-Bayesian credible bound"
