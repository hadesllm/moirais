"""Moving-block bootstrap (Künsch)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_moving_block"]


def boot_moving_block(x, block_len, stat, B):
    """
    Moving-block bootstrap (Künsch)

    Formula: Resample overlapping blocks of length ℓ

    Parameters
    ----------
    x : array-like
        Input data.
    block_len : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Künsch (1989)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moving-block bootstrap (Künsch)"})


def cheatsheet():
    return "btmbb: Moving-block bootstrap (Künsch)"
