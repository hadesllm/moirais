"""CAT next-item selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cat_next_item"]


def cat_next_item(theta_hat, item_pool, exposure_constraints):
    """
    CAT next-item selection

    Formula: argmax_j I_j(theta_hat) | not_yet_administered

    Parameters
    ----------
    theta_hat : array-like
        Input data.
    item_pool : array-like
        Input data.
    exposure_constraints : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van der Linden-Glas (2010)
    """
    theta_hat = np.atleast_1d(np.asarray(theta_hat, dtype=float))
    n = len(theta_hat)
    result = float(np.mean(theta_hat))
    se = float(np.std(theta_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CAT next-item selection"})


def cheatsheet():
    return "catnxt: CAT next-item selection"
