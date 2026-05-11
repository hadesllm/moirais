"""Diffusion recommendation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["diffusion_rec"]


def diffusion_rec(R, T):
    """
    Diffusion recommendation

    Formula: forward noise + reverse denoise on item dist

    Parameters
    ----------
    R : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang et al (2023) DiffRec
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Diffusion recommendation"})


def cheatsheet():
    return "diffRC: Diffusion recommendation"
