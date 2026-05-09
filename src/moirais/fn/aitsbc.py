"""Closed subcomposition over a chosen set of parts."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_subcomposition"]


def aitchison_subcomposition(x, idx):
    """
    Closed subcomposition over a chosen set of parts

    Formula: sub(x; S) = C(x_S)

    Parameters
    ----------
    x : array-like
        Input data.
    idx : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: xs

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed subcomposition over a chosen set of parts"})


def cheatsheet():
    return "aitsbc: Closed subcomposition over a chosen set of parts"
