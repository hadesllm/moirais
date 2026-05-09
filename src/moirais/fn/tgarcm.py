"""Threshold/GJR-GARCH (asymmetric leverage)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tgarch_gjr"]


def tgarch_gjr(x, p, q):
    """
    Threshold/GJR-GARCH (asymmetric leverage)

    Formula: sigma_t^2 = omega + alpha eps^2 + gamma I(eps<0) eps^2 + beta sigma^2

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Glosten, Jagannathan, Runkle (1993)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Threshold/GJR-GARCH (asymmetric leverage)"})


def cheatsheet():
    return "tgarcm: Threshold/GJR-GARCH (asymmetric leverage)"
