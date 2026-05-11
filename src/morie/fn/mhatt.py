# morie.fn — function file (hadesllm/morie)
r"""Multi-head attention (scaled dot-product).

Self-attention mechanism with multiple representation subspaces.

References
----------
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
Attention is all you need.
In NIPS (pp. 5998-6008).
"""

__all__ = ["mhatt"]

import numpy as np
from scipy.special import softmax
from ._richresult import RichResult


def mhatt(
    query,
    key,
    value,
    num_heads=8,
    d_model=512,
):
    """
    Multi-head scaled dot-product attention.

    Parameters
    ----------
    query : ndarray
        Query, shape (seq_len, d_model) or (batch, seq_len, d_model).
    key : ndarray
        Key, shape (seq_len, d_model).
    value : ndarray
        Value, shape (seq_len, d_model).
    num_heads : int, optional
        Number of attention heads. Default 8.
    d_model : int, optional
        Model dimension. Default 512.

    Returns
    -------
    dict
        Keys: 'output', 'attention_weights'.
    """
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)

    if d_model % num_heads != 0:
        raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")

    d_k = d_model // num_heads

    scores = np.dot(query, key.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)

    output = np.dot(attention_weights, value)

    return RichResult(payload={"output": output, "attention_weights": attention_weights})
