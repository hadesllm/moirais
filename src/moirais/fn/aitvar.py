"""Variation matrix of a compositional sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_variation"]


def aitchison_variation(X):
    """
    Variation matrix of a compositional sample

    Formula: T_{ij} = var(log(x_i/x_j))

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T

    References
    ----------
    Aitchison (1986)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variation matrix of a compositional sample"})


def cheatsheet():
    return "aitvar: Variation matrix of a compositional sample"
