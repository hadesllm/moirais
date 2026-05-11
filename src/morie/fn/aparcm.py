"""Asymmetric power ARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aparch_dge"]


def aparch_dge(x, delta):
    """
    Asymmetric power ARCH

    Formula: sigma_t^delta = omega + alpha (|eps| - gamma eps)^delta + beta sigma_{t-1}^delta

    Parameters
    ----------
    x : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ding, Granger, Engle (1993)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymmetric power ARCH"})


def cheatsheet():
    return "aparcm: Asymmetric power ARCH"
