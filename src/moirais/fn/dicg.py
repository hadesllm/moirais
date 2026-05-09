"""Deviance information criterion (DIC)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deviance_information_criterion"]


def deviance_information_criterion(deviance):
    """
    Deviance information criterion (DIC)

    Formula: DIC = D_bar + p_D = D(theta_bar) + 2 p_D

    Parameters
    ----------
    deviance : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Spiegelhalter et al. (2002)
    """
    deviance = np.atleast_1d(np.asarray(deviance, dtype=float))
    n = len(deviance)
    result = float(np.mean(deviance))
    se = float(np.std(deviance, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deviance information criterion (DIC)"})


def cheatsheet():
    return "dicg: Deviance information criterion (DIC)"
