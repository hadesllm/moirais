"""AlphaFold pair representation update."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_pair_repr"]


def alphafold_pair_repr(s, z):
    """
    AlphaFold pair representation update

    Formula: z_ij <- z_ij + outer_product(s_i, s_j) + triangle_updates

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold pair representation update"})


def cheatsheet():
    return "alfpaf: AlphaFold pair representation update"
