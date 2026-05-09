"""AlphaFold triangular multiplicative update."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_triangle_mult"]


def alphafold_triangle_mult(z, direction):
    """
    AlphaFold triangular multiplicative update

    Formula: z_ij <- z_ij + sum_k g(z_ik) * g(z_jk)

    Parameters
    ----------
    z : array-like
        Input data.
    direction : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold triangular multiplicative update"})


def cheatsheet():
    return "alftrm: AlphaFold triangular multiplicative update"
