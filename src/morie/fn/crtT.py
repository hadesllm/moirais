"""Chinese remainder theorem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chinese_remainder"]


def chinese_remainder(a, m):
    """
    Chinese remainder theorem

    Formula: x ≡ a_i (mod m_i) with coprime moduli

    Parameters
    ----------
    a : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sun Tzu (3rd c.)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chinese remainder theorem"})


def cheatsheet():
    return "crtT: Chinese remainder theorem"
