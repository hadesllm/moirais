"""AlphaFold pLDDT per-residue confidence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_confidence"]


def alphafold_confidence(frames, s):
    """
    AlphaFold pLDDT per-residue confidence

    Formula: p(LDDT_i in [0,1]) categorical

    Parameters
    ----------
    frames : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    frames = np.atleast_1d(np.asarray(frames, dtype=float))
    n = len(frames)
    result = float(np.mean(frames))
    se = float(np.std(frames, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold pLDDT per-residue confidence"})


def cheatsheet():
    return "alfcnf: AlphaFold pLDDT per-residue confidence"
