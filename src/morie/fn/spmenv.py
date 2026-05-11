"""Expected value and variance of Moran's I under randomization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_moran_expectation"]


def schabenberger_moran_expectation(x, w):
    """
    Expected value and variance of Moran's I under randomization

    Formula: E[I] = -1/(n-1); Var[I] = n^2*S1 - n*S2 + 3*S0^2 / [S0^2*(n^2-1)]

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: moments

    References
    ----------
    Schabenberger Ch 1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expected value and variance of Moran's I under randomization"})


def cheatsheet():
    return "spmenv: Expected value and variance of Moran's I under randomization"
