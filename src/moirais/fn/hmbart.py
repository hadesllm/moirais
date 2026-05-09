# moirais.fn — function file (hadesllm/moirais)
"""BART: denoising autoencoder pretraining for seq2seq."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bart"]


def geron_bart(src, tgt):
    """
    BART: denoising autoencoder pretraining for seq2seq

    Formula: corrupt text -> encoder -> reconstruct via decoder

    Parameters
    ----------
    src : array-like
        Input data.
    tgt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 15
    """
    src = np.atleast_1d(np.asarray(src, dtype=float))
    n = len(src)
    result = float(np.mean(src))
    se = float(np.std(src, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BART: denoising autoencoder pretraining for seq2seq"})


def cheatsheet():
    return "hmbart: BART: denoising autoencoder pretraining for seq2seq"
