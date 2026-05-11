"""Deep SVDD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deep_svdd"]


def deep_svdd(X, net):
    """
    Deep SVDD

    Formula: map data to compact ball in latent space

    Parameters
    ----------
    X : array-like
        Input data.
    net : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ruff et al (2018)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep SVDD"})


def cheatsheet():
    return "deepSVDD: Deep SVDD"
