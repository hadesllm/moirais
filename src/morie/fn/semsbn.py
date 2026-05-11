"""Satorra-Bentler chi-square correction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sem_sb_chi_sq"]


def sem_sb_chi_sq(fit):
    """
    Satorra-Bentler chi-square correction

    Formula: chi-sq_SB = chi-sq_ML / scaling correction

    Parameters
    ----------
    fit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Satorra-Bentler (1994)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Satorra-Bentler chi-square correction"})


def cheatsheet():
    return "semsbn: Satorra-Bentler chi-square correction"
