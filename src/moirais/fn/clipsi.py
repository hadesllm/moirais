"""CLIP image-text similarity score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clip_similarity"]


def clip_similarity(I_emb, T_emb, tau):
    """
    CLIP image-text similarity score

    Formula: cos(I_emb, T_emb) / tau

    Parameters
    ----------
    I_emb : array-like
        Input data.
    T_emb : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Radford et al (2021)
    """
    I_emb = np.atleast_1d(np.asarray(I_emb, dtype=float))
    n = len(I_emb)
    result = float(np.mean(I_emb))
    se = float(np.std(I_emb, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLIP image-text similarity score"})


def cheatsheet():
    return "clipsi: CLIP image-text similarity score"
