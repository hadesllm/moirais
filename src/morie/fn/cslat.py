# morie.fn -- function file (hadesllm/morie)
"""Causal (autoregressive) attention mask (Radford et al. 2019)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_attention_mask"]


def causal_attention_mask(x):
    """Build the lower-triangular causal mask.

    Formula:  mask[i, j] = -inf if j > i else 0.

    Parameters
    ----------
    x : int OR array-like
        If int, treat as sequence length ``n``.  If array-like, use
        ``len(x)`` as the sequence length.

    Returns
    -------
    RichResult with keys: tensor (mask), n.
    """
    if np.isscalar(x):
        n = int(x)
    else:
        arr = np.asarray(x)
        n = arr.shape[-2] if arr.ndim >= 2 else arr.shape[-1]
    mask = np.zeros((n, n), dtype=float)
    mask[np.triu_indices(n, k=1)] = -np.inf
    return RichResult(
        title="Causal Attention Mask (Radford 2019)",
        summary_lines=[("seq_len", n),
                       ("allowed", int(n * (n + 1) // 2))],
        payload={"tensor": mask, "n": n, "method": "causal-mask"},
    )


def cheatsheet():
    return "cslat(n): lower-triangular causal mask, 0 on/below, -inf above"


# CANONICAL TEST
# >>> m = causal_attention_mask(3)["tensor"]
# >>> np.array_equal(np.isinf(m), np.array(
# ...     [[False, True, True], [False, False, True], [False, False, False]]))
# True
