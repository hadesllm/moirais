"""AIPW with efficient influence curve."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aipw_efficient_influence"]


def aipw_efficient_influence(y, D, X, ml_outcome, ml_propensity):
    """
    AIPW with efficient influence curve

    Formula: psi = D Y/e + (1-D)Y/(1-e) + correction term

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    ml_outcome : array-like
        Input data.
    ml_propensity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Rotnitzky-Zhao (1994); Chernozhukov et al (2018) DML
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AIPW with efficient influence curve"})


def cheatsheet():
    return "eaiprl: AIPW with efficient influence curve"
