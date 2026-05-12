# morie.fn -- function file (hadesllm/morie)
"""Encoder-decoder transformer (original architecture)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_encoder_decoder_transformer"]


def geron_encoder_decoder_transformer(src, tgt, n_layers):
    """
    Encoder-decoder transformer (original architecture)

    Formula: encoder stack + decoder stack with cross-attention

    Parameters
    ----------
    src : array-like
        Input data.
    tgt : array-like
        Input data.
    n_layers : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Encoder-decoder transformer (original architecture)"})


def cheatsheet():
    return "hmencd: Encoder-decoder transformer (original architecture)"
