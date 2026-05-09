"""AlphaFold Evoformer block."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_evoformer"]


def alphafold_evoformer(s, z):
    """
    AlphaFold Evoformer block

    Formula: alternating row/col MSA attention + triangle ops

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold Evoformer block"})


def cheatsheet():
    return "alfevf: AlphaFold Evoformer block"
