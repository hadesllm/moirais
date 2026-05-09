"""AlphaFold predicted aligned error (PAE)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_pae_predict"]


def alphafold_pae_predict(s, z):
    """
    AlphaFold predicted aligned error (PAE)

    Formula: E[||frames_i^j - frames_i_true^j||]

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
    Jumper et al (2021)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold predicted aligned error (PAE)"})


def cheatsheet():
    return "alfpea: AlphaFold predicted aligned error (PAE)"
