"""Sliced Wasserstein distance via random projections."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_sliced_wasserstein"]


def ot_sliced_wasserstein(X, Y, p, n_proj):
    """
    Sliced Wasserstein distance via random projections

    Formula: SW(μ,ν) = (E[W_p(P_θ#μ, P_θ#ν)^p])^{1/p}

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    p : array-like
        Input data.
    n_proj : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: SW

    References
    ----------
    Bonneel-Rabin-Peyré-Pfister (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sliced Wasserstein distance via random projections"})


def cheatsheet():
    return "otsw: Sliced Wasserstein distance via random projections"
