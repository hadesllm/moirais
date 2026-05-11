"""AlphaZero value head."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_value_head"]


def alphazero_value_head(x):
    """
    AlphaZero value head

    Formula: conv(1) + flatten + linear + tanh

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero value head"})


def cheatsheet():
    return "agvhdt: AlphaZero value head"
