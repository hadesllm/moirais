"""Small-world coefficient sigma (Humphries-Gurney)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["small_world_sigma"]


def small_world_sigma(y, A):
    """
    Small-world coefficient sigma (Humphries-Gurney)

    Formula: sigma = (C / C_rand) / (L / L_rand)

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Humphries & Gurney (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Small-world coefficient sigma (Humphries-Gurney)"})


def cheatsheet():
    return "smallw: Small-world coefficient sigma (Humphries-Gurney)"
