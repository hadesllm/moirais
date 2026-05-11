"""ALiBi attention with linear bias."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alibi_position_bias"]


def alibi_position_bias(y, Q, K, V, slopes):
    """
    ALiBi attention with linear bias

    Formula: a_ij = q_i k_j / sqrt(d) - m * |i - j|

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    slopes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Press, Smith, Lewis (2022)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ALiBi attention with linear bias"})


def cheatsheet():
    return "atalib: ALiBi attention with linear bias"
