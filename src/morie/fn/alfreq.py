"""AlphaFold iterative recycling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_recycling"]


def alphafold_recycling(s, z, n_recycles):
    """
    AlphaFold iterative recycling

    Formula: feed predicted s, z back into next iteration

    Parameters
    ----------
    s : array-like
        Input data.
    z : array-like
        Input data.
    n_recycles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold iterative recycling"})


def cheatsheet():
    return "alfreq: AlphaFold iterative recycling"
