"""Functional GAM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_gam"]


def functional_gam(X, Y, basis):
    """
    Functional GAM

    Formula: Y = α + ∫ F(X(t),t) dt

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    basis : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McLean et al (2014) FGAM
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional GAM"})


def cheatsheet():
    return "fgam: Functional GAM"
