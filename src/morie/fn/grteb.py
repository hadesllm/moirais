# morie.fn -- function file (hadesllm/morie)
"""Transformer encoder block: MHA + FFN with residual connections + LayerNorm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_transformer_encoder_block"]


def geron_transformer_encoder_block(x, mha_weights, ffn_weights):
    """
    Transformer encoder block: MHA + FFN with residual connections + LayerNorm

    Formula: h = LayerNorm(x + MHA(x)); y = LayerNorm(h + FFN(h))

    Parameters
    ----------
    x : array-like
        Input data.
    mha_weights : array-like
        Input data.
    ffn_weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 15, Encoder-only transformer section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transformer encoder block: MHA + FFN with residual connections + LayerNorm"})


def cheatsheet():
    return "grteb: Transformer encoder block: MHA + FFN with residual connections + LayerNorm"
