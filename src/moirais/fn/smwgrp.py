"""Small-world coefficient sigma."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["small_worldness"]


def small_worldness(G, random_baseline):
    """
    Small-world coefficient sigma

    Formula: sigma = (C/C_rand) / (L/L_rand)

    Parameters
    ----------
    G : array-like
        Input data.
    random_baseline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Humphries-Gurney (2008)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Small-world coefficient sigma"})


def cheatsheet():
    return "smwgrp: Small-world coefficient sigma"
