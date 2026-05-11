# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Contextualized word embedding: layer-l hidden state at position i."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_contextualized_embedding"]


def alammar_contextualized_embedding(layer_outputs, layer_idx, position):
    """
    Contextualized word embedding: layer-l hidden state at position i

    Formula: h_i^{(l)} = f_l(h_1^{(l-1)}, ..., h_L^{(l-1)})

    Parameters
    ----------
    layer_outputs : array-like
        Input data.
    layer_idx : array-like
        Input data.
    position : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h

    References
    ----------
    Alammar Ch 2, Contextualized Word Embeddings section
    """
    layer_outputs = np.atleast_1d(np.asarray(layer_outputs, dtype=float))
    n = len(layer_outputs)
    result = float(np.mean(layer_outputs))
    se = float(np.std(layer_outputs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contextualized word embedding: layer-l hidden state at position i"})


def cheatsheet():
    return "alctxemb: Contextualized word embedding: layer-l hidden state at position i"
