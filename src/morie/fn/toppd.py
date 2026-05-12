# morie.fn -- function file (hadesllm/morie)
"""Nucleus (top-p) sampling (Holtzman et al. 2020)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["top_p_nucleus"]


def top_p_nucleus(x, p: float = 0.9, T: float = 1.0):
    """Top-p (nucleus) probability filtering.

    Formula: rank tokens by probability; keep the smallest set whose
    cumulative probability >= ``p``; mask the rest, renormalise.

    Parameters
    ----------
    x : array-like of logits, shape (V,).
    p : float, in (0, 1]
        Cumulative-probability cutoff.
    T : float
        Temperature applied first.

    Returns
    -------
    RichResult with keys: tensor (probabilities), keep_mask,
    n_kept.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    z = np.asarray(x, dtype=float).ravel() / T
    z = z - np.max(z)
    probs = np.exp(z); probs = probs / np.sum(probs)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cum, p)) + 1
    cutoff = max(1, min(cutoff, probs.size))
    keep_sorted = np.zeros_like(probs, dtype=bool)
    keep_sorted[:cutoff] = True
    keep = np.zeros_like(probs, dtype=bool)
    keep[order] = keep_sorted
    filtered = np.where(keep, probs, 0.0)
    filtered = filtered / filtered.sum()
    return RichResult(
        title="Top-p Nucleus Sampling (Holtzman 2020)",
        summary_lines=[("p", p),
                       ("n_kept", int(keep.sum())),
                       ("V", probs.size)],
        payload={"tensor": filtered, "keep_mask": keep,
                 "n_kept": int(keep.sum()), "p": p,
                 "method": "top-p"},
    )


def cheatsheet():
    return "toppd(logits, p): nucleus (top-p) filtered softmax"


# CANONICAL TEST
# >>> r = top_p_nucleus([0.0, 0.0, 5.0], p=0.5)
# >>> int(r["n_kept"])
# 1
