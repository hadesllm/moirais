# morie.fn -- function file (hadesllm/morie)
"""RLHF reward-model score (Ouyang et al. 2022, InstructGPT)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rlhf_reward"]


def rlhf_reward(x, w=None, b: float = 0.0):
    """Linear-head reward model on top of last-token hidden states.

    Formula:  r(x, y) = w^T h_last(x, y) + b.

    Parameters
    ----------
    x : ndarray, shape (..., d_hidden)
        Last-token hidden state(s) of the policy's response.
    w : ndarray, shape (d_hidden,), optional
        Reward-head weights.  Defaults to mean-pooling (1/d * ones).
    b : float
        Reward-head bias.

    Returns
    -------
    RichResult with keys: value (scalar headline reward), tensor
    (per-sample rewards), w, b.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    d = x.shape[-1]
    if w is None:
        w = np.ones(d) / d
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size != d:
        raise ValueError(f"w must have length {d}")
    r = x @ w + b
    val = float(r[0]) if r.size > 0 else float("nan")
    return RichResult(
        title="RLHF Reward Score (Ouyang 2022)",
        summary_lines=[("mean_reward", float(np.mean(r))),
                       ("n", int(r.size))],
        payload={"value": val, "tensor": r, "w": w, "b": b,
                 "method": "rlhf-reward-head"},
    )


def cheatsheet():
    return "rlhfd(h_last, w, b): linear reward head r = w^T h + b"


# CANONICAL TEST
# >>> r = rlhf_reward(np.array([[1.0, 1.0]]), w=np.array([0.5, 0.5]), b=0.0)
# >>> float(r["value"])
# 1.0
