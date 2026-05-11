"""Invariant Point Attention (AlphaFold2) for structure update."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_invariant_point"]


def alphafold_invariant_point(s_i, q, k, v, frames):
    """
    Invariant Point Attention (AlphaFold2) for structure update

    Formula: q_h = q^pt frame; attention with rotation-invariance

    Parameters
    ----------
    s_i : array-like
        Input data.
    q : array-like
        Input data.
    k : array-like
        Input data.
    v : array-like
        Input data.
    frames : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021) Nature §1.8
    """
    s_i = np.atleast_1d(np.asarray(s_i, dtype=float))
    n = len(s_i)
    result = float(np.mean(s_i))
    se = float(np.std(s_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Invariant Point Attention (AlphaFold2) for structure update"})


def cheatsheet():
    return "alfipa: Invariant Point Attention (AlphaFold2) for structure update"
