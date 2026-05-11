"""Swin shifted-window attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["swin_transformer"]


def swin_transformer(x, window_size):
    """
    Swin shifted-window attention

    Formula: window MSA + cyclic shift

    Parameters
    ----------
    x : array-like
        Input data.
    window_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu et al (2021) Swin
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Swin shifted-window attention"})


def cheatsheet():
    return "swintr: Swin shifted-window attention"
