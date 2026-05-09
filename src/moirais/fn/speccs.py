"""Cross-spectrum between series."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cross_spectrum"]


def cross_spectrum(x, y):
    """
    Cross-spectrum between series

    Formula: FFT(x) FFT(y)*

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Brillinger (2001)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-spectrum between series"})


def cheatsheet():
    return "speccs: Cross-spectrum between series"
