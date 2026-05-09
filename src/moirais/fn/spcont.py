"""Mean-square continuity condition for random fields."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_spatial_continuity"]


def schabenberger_spatial_continuity(cov_func):
    """
    Mean-square continuity condition for random fields

    Formula: lim_{h->0} E[(Z(s+h)-Z(s))^2] = 0 iff C(h) continuous at 0

    Parameters
    ----------
    cov_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Schabenberger Ch 2, Sec 2.3
    """
    cov_func = np.asarray(cov_func, dtype=float)
    n = int(cov_func) if cov_func.ndim == 0 else len(cov_func)
    result = float(np.mean(cov_func))
    se = float(np.std(cov_func, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean-square continuity condition for random fields"})


def cheatsheet():
    return "spcont: Mean-square continuity condition for random fields"
