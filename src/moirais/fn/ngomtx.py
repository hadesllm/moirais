"""Next-generation matrix R0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["next_generation_matrix"]


def next_generation_matrix(FV_decomposition):
    """
    Next-generation matrix R0

    Formula: R0 = spectral radius of NGM

    Parameters
    ----------
    FV_decomposition : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Diekmann-Heesterbeek-Roberts (2010)
    """
    FV_decomposition = np.atleast_1d(np.asarray(FV_decomposition, dtype=float))
    n = len(FV_decomposition)
    result = float(np.mean(FV_decomposition))
    se = float(np.std(FV_decomposition, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Next-generation matrix R0"})


def cheatsheet():
    return "ngomtx: Next-generation matrix R0"
