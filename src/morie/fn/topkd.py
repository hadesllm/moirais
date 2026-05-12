# morie.fn — function file (hadesllm/morie)
"""Top-k decoding (Fan et al. 2018)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["top_k_decoding"]


def top_k_decoding(x, k: int = 5, T: float = 1.0):
    """Top-k probability filtering.

    Formula: keep the ``k`` highest-probability tokens, mask the rest
    to zero, renormalise so the remaining mass sums to 1.

    Parameters
    ----------
    x : array-like of logits, shape (..., V).
    k : int
        Top-k cutoff.
    T : float
        Optional temperature applied before the top-k filter.

    Returns
    -------
    RichResult with keys: tensor (probabilities), topk_indices,
    topk_logits.
    """
    z = np.asarray(x, dtype=float) / T
    V = z.shape[-1]
    k = max(1, min(int(k), V))
    part = np.partition(z, -k, axis=-1)
    thresh = part[..., -k:].min(axis=-1, keepdims=True)
    mask = z >= thresh
    z_filtered = np.where(mask, z, -np.inf)
    z_filtered = z_filtered - np.max(z_filtered, axis=-1, keepdims=True)
    e = np.exp(z_filtered)
    p = e / np.sum(e, axis=-1, keepdims=True)
    topk_idx = np.argsort(z, axis=-1)[..., -k:][..., ::-1]
    topk_logits = np.take_along_axis(z, topk_idx, axis=-1)
    return RichResult(
        title="Top-k Decoding (Fan 2018)",
        summary_lines=[("k", k), ("V", V)],
        payload={"tensor": p, "topk_indices": topk_idx,
                 "topk_logits": topk_logits, "k": k,
                 "method": "top-k"},
    )


def cheatsheet():
    return "topkd(logits, k): top-k filtered softmax"


# CANONICAL TEST
# >>> r = top_k_decoding([1.0, 2.0, 3.0, 4.0, 5.0], k=2)
# >>> int((r["tensor"] > 0).sum())
# 2
