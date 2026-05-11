"""Shannon entropy of a discrete distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shannon_entropy"]


def shannon_entropy(y, base):
    """
    Shannon entropy of a discrete distribution

    Formula: H(X) = -sum_x p(x) log p(x)

    Parameters
    ----------
    y : array-like
        Input data.
    base : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shannon (1948)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shannon entropy of a discrete distribution"})


def cheatsheet():
    return "shanen: Shannon entropy of a discrete distribution"
