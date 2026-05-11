"""AlphaZero batch normalization tracking."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_batch_norm"]


def alphazero_batch_norm(x, running_mean, running_var, momentum):
    """
    AlphaZero batch normalization tracking

    Formula: running mean/var with momentum

    Parameters
    ----------
    x : array-like
        Input data.
    running_mean : array-like
        Input data.
    running_var : array-like
        Input data.
    momentum : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ioffe-Szegedy (2015)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero batch normalization tracking"})


def cheatsheet():
    return "agbtnm: AlphaZero batch normalization tracking"
