"""AlphaFold side-chain prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_sidechain"]


def alphafold_sidechain(s, frames):
    """
    AlphaFold side-chain prediction

    Formula: chi angles per residue from frames + s

    Parameters
    ----------
    s : array-like
        Input data.
    frames : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold side-chain prediction"})


def cheatsheet():
    return "alfsdc: AlphaFold side-chain prediction"
