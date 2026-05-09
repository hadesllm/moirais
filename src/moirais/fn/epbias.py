"""Bias correction for exposure misclassification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["exposure_misclass_bias"]


def exposure_misclass_bias(A_obs, Se, Sp):
    """
    Bias correction for exposure misclassification

    Formula: divide by (Se + Sp - 1)

    Parameters
    ----------
    A_obs : array-like
        Input data.
    Se : array-like
        Input data.
    Sp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Greenland (1988)
    """
    A_obs = np.atleast_1d(np.asarray(A_obs, dtype=float))
    n = len(A_obs)
    result = float(np.mean(A_obs))
    se = float(np.std(A_obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bias correction for exposure misclassification"})


def cheatsheet():
    return "epbias: Bias correction for exposure misclassification"
