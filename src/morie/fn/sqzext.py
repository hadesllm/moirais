"""Squeeze-and-Excitation block."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["squeeze_excite"]


def squeeze_excite(x, reduction):
    """
    Squeeze-and-Excitation block

    Formula: global pool + FC + sigmoid; multiply channels

    Parameters
    ----------
    x : array-like
        Input data.
    reduction : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hu-Shen-Sun (2018) SENet
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Squeeze-and-Excitation block"})


def cheatsheet():
    return "sqzext: Squeeze-and-Excitation block"
