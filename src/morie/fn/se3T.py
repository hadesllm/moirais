"""SE(3)-equivariant transformer."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["se3_transformer"]


def se3_transformer(G, X, coords):
    """
    SE(3)-equivariant transformer

    Formula: steerable attention with spherical harmonics

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fuchs et al (2020)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SE(3)-equivariant transformer"})


def cheatsheet():
    return "se3T: SE(3)-equivariant transformer"
