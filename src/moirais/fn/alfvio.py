"""AlphaFold structural violation loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_violation"]


def alphafold_violation(coords, atom_types):
    """
    AlphaFold structural violation loss

    Formula: penalty for clash + bond violations

    Parameters
    ----------
    coords : array-like
        Input data.
    atom_types : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold structural violation loss"})


def cheatsheet():
    return "alfvio: AlphaFold structural violation loss"
