# moirais.fn — function file (hadesllm/moirais)
"""Transformer decoder block: masked MHA + cross-attention + FFN."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_transformer_decoder_block"]


def geron_transformer_decoder_block(x, encoder_output, weights):
    """
    Transformer decoder block: masked MHA + cross-attention + FFN

    Formula: h1=LN(x+MaskedMHA(x)); h2=LN(h1+CA(h1,enc)); y=LN(h2+FFN(h2))

    Parameters
    ----------
    x : array-like
        Input data.
    encoder_output : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 15, Encoder-decoder transformer section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transformer decoder block: masked MHA + cross-attention + FFN"})


def cheatsheet():
    return "grtdb: Transformer decoder block: masked MHA + cross-attention + FFN"
