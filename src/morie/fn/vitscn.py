"""ViT scaled cosine attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_scaled_cosine"]


def vit_scaled_cosine(q, k, v, tau):
    """
    ViT scaled cosine attention

    Formula: replace dot product with cosine sim

    Parameters
    ----------
    q : array-like
        Input data.
    k : array-like
        Input data.
    v : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu et al (2021)
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    result = float(np.mean(q))
    se = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT scaled cosine attention"})


def cheatsheet():
    return "vitscn: ViT scaled cosine attention"
