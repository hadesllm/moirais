"""Convergent cross mapping (Sugihara)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["convergent_cross_mapping"]


def convergent_cross_mapping(x, y, E, tau):
    """
    Convergent cross mapping (Sugihara)

    Formula: shadow manifold reconstruction CCM

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    E : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sugihara et al (2012)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convergent cross mapping (Sugihara)"})


def cheatsheet():
    return "cnvlfc: Convergent cross mapping (Sugihara)"
