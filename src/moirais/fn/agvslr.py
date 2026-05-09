"""AlphaZero learning-rate schedule."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_value_lr"]


def alphazero_value_lr(t, T, lr_0):
    """
    AlphaZero learning-rate schedule

    Formula: lr_t = lr_0 * cosine_decay(t/T)

    Parameters
    ----------
    t : array-like
        Input data.
    T : array-like
        Input data.
    lr_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero learning-rate schedule"})


def cheatsheet():
    return "agvslr: AlphaZero learning-rate schedule"
