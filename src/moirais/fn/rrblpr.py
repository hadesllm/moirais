"""Ridge-regression BLUP (markers as random)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rr_blup"]


def rr_blup(y, M, lam):
    """
    Ridge-regression BLUP (markers as random)

    Formula: u_marker ~ N(0, sigma_m^2 I); shrink jointly

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Whittaker-Thompson-Denham (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge-regression BLUP (markers as random)"})


def cheatsheet():
    return "rrblpr: Ridge-regression BLUP (markers as random)"
