# morie.fn — function file (hadesllm/morie)
r"""KV-cache for autoregressive generation.

Stores key/value tensors to avoid recomputation.

References
----------
Pope, R., Douglas, S., Chowdhery, A., Devries, M., Hall, J., Levin, N., ... & Norouzi, M. (2022).
Efficiently scaling transformer inference.
arXiv preprint arXiv:2211.05102.
"""

__all__ = ["kvcch"]

import numpy as np


def kvcch(
    key,
    value,
    cache_k=None,
    cache_v=None,
):
    """
    KV-cache update for autoregressive generation.

    Parameters
    ----------
    key : ndarray
        New key tokens, shape (batch, 1, d_k) or (seq_len, d_k).
    value : ndarray
        New value tokens, shape (batch, 1, d_v) or (seq_len, d_v).
    cache_k : ndarray, optional
        Cached keys from previous steps.
    cache_v : ndarray, optional
        Cached values from previous steps.

    Returns
    -------
    dict
        Keys: 'cache_k', 'cache_v', 'full_k', 'full_v'.
    """
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)

    if cache_k is None:
        cache_k = key
    else:
        cache_k = np.concatenate([cache_k, key], axis=0)

    if cache_v is None:
        cache_v = value
    else:
        cache_v = np.concatenate([cache_v, value], axis=0)

    return {
        "cache_k": cache_k,
        "cache_v": cache_v,
        "full_k": cache_k,
        "full_v": cache_v,
    }
