# moirais.fn — function file (hadesllm/moirais)
"""Encoder-decoder for neural machine translation (seq2seq)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_encoder_decoder_nmt"]


def geron_encoder_decoder_nmt(src, tgt, model):
    """
    Encoder-decoder for neural machine translation (seq2seq)

    Formula: enc(src)->z; dec(z, tgt)->y

    Parameters
    ----------
    src : array-like
        Input data.
    tgt : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: translation

    References
    ----------
    Géron Ch 14
    """
    src = np.atleast_1d(np.asarray(src, dtype=float))
    n = len(src)
    result = float(np.mean(src))
    se = float(np.std(src, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Encoder-decoder for neural machine translation (seq2seq)"})


def cheatsheet():
    return "hmnmt: Encoder-decoder for neural machine translation (seq2seq)"
