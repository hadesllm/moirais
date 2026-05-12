# morie.fn — function file (hadesllm/morie)
"""Sparse attention pattern (Child et al. 2019, Sparse Transformer)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["sparse_attention"]


def sparse_attention(x, window: int = 4, stride: int = 8,
                     n_random: int = 0, seed: int = 0):
    """Build a Child-2019 sparse attention mask: sliding window +
    strided global tokens + optional random connections.

    Formula:  mask[i, j] = 1 if any of
                            - |i - j| <= window               (local)
                            - j == 0  or  j % stride == 0     (strided global)
                            - random selection                (n_random per row)
              else 0.

    Complement positions are set to ``-inf`` (additive softmax mask).

    Parameters
    ----------
    x : int OR array-like
        Either a sequence length, or any array whose last-but-one axis
        gives the sequence length.
    window : int
        Half-width of the local sliding window.
    stride : int
        Period of the strided global connections.
    n_random : int
        Number of random connections per query row.
    seed : int
        RNG seed for the random pattern.

    Returns
    -------
    RichResult with keys: tensor (additive mask, -inf or 0),
    boolean (boolean attend-mask), density.
    """
    if np.isscalar(x):
        N = int(x)
    else:
        arr = np.asarray(x)
        N = arr.shape[-2] if arr.ndim >= 2 else arr.shape[-1]
    rng = np.random.default_rng(seed)
    M = np.zeros((N, N), dtype=bool)
    for i in range(N):
        lo = max(0, i - window); hi = min(N, i + window + 1)
        M[i, lo:hi] = True
        M[i, ::stride] = True
        if n_random > 0:
            picks = rng.choice(N, size=min(n_random, N), replace=False)
            M[i, picks] = True
    additive = np.where(M, 0.0, -np.inf)
    density = float(M.sum() / (N * N))
    return RichResult(
        title="Sparse Attention (Child 2019)",
        summary_lines=[("N", N), ("window", window), ("stride", stride),
                       ("density", density)],
        payload={"tensor": additive, "boolean": M, "density": density,
                 "method": "sparse-attention"},
    )


def cheatsheet():
    return "spqkv(N, window, stride, n_random): Child-2019 sparse mask"


# CANONICAL TEST
# >>> r = sparse_attention(8, window=1, stride=4, n_random=0)
# >>> bool(r["boolean"][3, 3])
# True
