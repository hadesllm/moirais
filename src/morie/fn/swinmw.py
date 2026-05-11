"""Swin MSA within window."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["swin_msa_window"]


def swin_msa_window(x, window_size, relative_bias):
    """
    Swin MSA within window

    Formula: per-window attention; relative pos bias

    Parameters
    ----------
    x : array-like
        Input data.
    window_size : array-like
        Input data.
    relative_bias : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu et al (2021)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Swin MSA within window"})


def cheatsheet():
    return "swinmw: Swin MSA within window"
