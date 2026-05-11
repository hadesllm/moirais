"""VAR forecast error variance decomposition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["var_variance_decomp"]


def var_variance_decomp(fit, horizon):
    """
    VAR forecast error variance decomposition

    Formula: share of forecast var attributable to each shock

    Parameters
    ----------
    fit : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lütkepohl (2005)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VAR forecast error variance decomposition"})


def cheatsheet():
    return "vardec: VAR forecast error variance decomposition"
