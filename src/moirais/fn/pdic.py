"""Effective parameters from DIC (p_D)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_parameters_dic"]


def effective_parameters_dic(deviance):
    """
    Effective parameters from DIC (p_D)

    Formula: p_D = D_bar - D(theta_bar)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective parameters from DIC (p_D)"})


def cheatsheet():
    return "pdic: Effective parameters from DIC (p_D)"
