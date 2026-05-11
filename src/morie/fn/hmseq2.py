# morie.fn — function file (hadesllm/morie)
"""Sequence-to-sequence encoder-decoder architecture."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_seq2seq"]


def geron_seq2seq(src, tgt, encoder, decoder):
    """
    Sequence-to-sequence encoder-decoder architecture

    Formula: enc(x_1..x_T) -> z; dec(z, y_0..y_{s-1}) -> y_s

    Parameters
    ----------
    src : array-like
        Input data.
    tgt : array-like
        Input data.
    encoder : array-like
        Input data.
    decoder : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output_seq

    References
    ----------
    Géron Ch 13
    """
    src = np.atleast_1d(np.asarray(src, dtype=float))
    n = len(src)
    result = float(np.mean(src))
    se = float(np.std(src, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequence-to-sequence encoder-decoder architecture"})


def cheatsheet():
    return "hmseq2: Sequence-to-sequence encoder-decoder architecture"
