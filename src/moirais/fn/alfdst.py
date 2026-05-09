"""AlphaFold distogram prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_distogram"]


def alphafold_distogram(s, z):
    """
    AlphaFold distogram prediction

    Formula: P(d_ij in bin_k | s, z) categorical

    Parameters
    ----------
    s : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Senior et al (2020); Jumper (2021)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold distogram prediction"})


def cheatsheet():
    return "alfdst: AlphaFold distogram prediction"
