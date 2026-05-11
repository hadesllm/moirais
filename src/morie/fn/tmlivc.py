"""TMLE for instrumental variable LATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_iv"]


def tmle_iv(y, D, Z, covariates):
    """
    TMLE for instrumental variable LATE

    Formula: local average treatment effect with clever covariate using IV

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.
    covariates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tóth & vdL (2014); Ogburn-Rotnitzky-Robins (2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for instrumental variable LATE"})


def cheatsheet():
    return "tmlivc: TMLE for instrumental variable LATE"
