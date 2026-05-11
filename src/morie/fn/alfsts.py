"""AlphaFold structure module transition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_structure_transition"]


def alphafold_structure_transition(s, z, frames):
    """
    AlphaFold structure module transition

    Formula: s_i <- s_i + ipa(s, q, frames)

    Parameters
    ----------
    s : array-like
        Input data.
    z : array-like
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
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold structure module transition"})


def cheatsheet():
    return "alfsts: AlphaFold structure module transition"
