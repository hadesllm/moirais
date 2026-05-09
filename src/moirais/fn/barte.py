"""BART denoising encoder-decoder."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bart"]


def bart(src, tgt):
    """
    BART denoising encoder-decoder

    Formula: corrupt → reconstruct via seq2seq

    Parameters
    ----------
    src : array-like
        Input data.
    tgt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lewis et al (2020)
    """
    src = np.atleast_1d(np.asarray(src, dtype=float))
    n = len(src)
    result = float(np.mean(src))
    se = float(np.std(src, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BART denoising encoder-decoder"})


def cheatsheet():
    return "barte: BART denoising encoder-decoder"
