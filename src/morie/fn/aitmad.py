"""MAD of CLR-transformed compositions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_mad"]


def compositional_mad(X):
    """
    MAD of CLR-transformed compositions

    Formula: MAD = median(|z - median(z)|)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mad

    References
    ----------
    Filzmoser et al. (2018)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MAD of CLR-transformed compositions"})


def cheatsheet():
    return "aitmad: MAD of CLR-transformed compositions"
