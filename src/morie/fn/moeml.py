# morie.fn -- function file (hadesllm/morie)
"""Mixture-of-Experts gating (Shazeer et al. 2017)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["mixture_of_experts"]


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def mixture_of_experts(x, W_gate=None, experts=None, top_k: int = 2):
    """Sparsely-gated MoE forward.

    Formula:
        g(x)         = softmax(x W_g)              # (B, n_experts)
        topk_g(x)    = top-k mask of g(x), renormalised
        y            = sum_i  topk_g_i(x) * E_i(x)

    Parameters
    ----------
    x : ndarray, shape (B, d_in)
    W_gate : ndarray, shape (d_in, n_experts)
        Router weights.
    experts : list of (W_i, b_i) tuples, len == n_experts
        Each expert is an affine map y_i = x @ W_i + b_i with
        W_i shape (d_in, d_out).  If None, identity experts of
        (d_out == d_in) are used.
    top_k : int
        How many experts to route per token (Shazeer 2017 used 2).

    Returns
    -------
    RichResult with keys: tensor (output), gate (sparse weights),
    topk_idx, load (per-expert utilisation).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    B, d_in = x.shape
    if W_gate is None:
        n_experts = 2
        W_gate = np.zeros((d_in, n_experts))
    W_gate = np.asarray(W_gate, dtype=float)
    n_experts = W_gate.shape[1]
    if experts is None:
        experts = [(np.eye(d_in), np.zeros(d_in)) for _ in range(n_experts)]
    gate_logits = x @ W_gate
    gate = _softmax(gate_logits, axis=-1)
    k = max(1, min(int(top_k), n_experts))
    topk_idx = np.argsort(-gate, axis=-1)[:, :k]
    sparse = np.zeros_like(gate)
    np.put_along_axis(sparse, topk_idx,
                      np.take_along_axis(gate, topk_idx, axis=-1), axis=-1)
    sparse = sparse / np.sum(sparse, axis=-1, keepdims=True)
    expert_outs = [x @ np.asarray(W_i) + np.asarray(b_i)
                   for W_i, b_i in experts]
    E = np.stack(expert_outs, axis=0)  # (n_experts, B, d_out)
    y = np.einsum("be,ebd->bd", sparse, E)
    load = sparse.sum(axis=0) / B
    return RichResult(
        title="Sparsely-Gated MoE (Shazeer 2017)",
        summary_lines=[("n_experts", n_experts), ("top_k", k),
                       ("batch", B)],
        payload={"tensor": y, "gate": sparse, "topk_idx": topk_idx,
                 "load": load, "method": "MoE"},
    )


def cheatsheet():
    return "moeml(x, W_gate, experts, top_k): mixture of experts"


# CANONICAL TEST
# >>> x = np.array([[1.0, 0.0]])
# >>> W_gate = np.array([[10.0, -10.0], [0.0, 0.0]])
# >>> r = mixture_of_experts(x, W_gate=W_gate, top_k=1)
# >>> int(r["topk_idx"][0, 0])
# 0
