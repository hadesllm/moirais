# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Scaled dot-product attention mechanism."""

import numpy as np

from ._containers import DescriptiveResult


def attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V.

    :param Q: (n, d_k) query matrix.
    :param K: (m, d_k) key matrix.
    :param V: (m, d_v) value matrix.
    :param mask: (n, m) optional mask (True = keep, False = mask out).
    :return: DescriptiveResult with output, attention weights.

    References
    ----------
    Vaswani A et al. (2017). Attention Is All You Need. NeurIPS.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q[None, :]
    if K.ndim == 1:
        K = K[None, :]
    if V.ndim == 1:
        V = V[None, :]
    d_k = K.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    output = weights @ V

    return DescriptiveResult(
        name="attention",
        value=float(np.mean(weights.max(axis=-1))),
        extra={
            "output": output,
            "attention_weights": weights,
            "d_k": d_k,
            "n_queries": Q.shape[0],
            "n_keys": K.shape[0],
        },
    )


def cheatsheet() -> str:
    return "attention({}) -> Scaled dot-product attention mechanism."
