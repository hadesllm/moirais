"""Transformer (single-head self-attention) for genomic prediction -- NumPy.

Forward-only deterministic attention + ridge-regression head. We do not
train the attention weights -- keeping them random projections -- and fit
the linear head by closed-form ridge on the pooled context vector. This
gives a NumPy-only, reproducible, sub-second implementation while still
returning the per-position attention map for inspection.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["transformer_genomic"]


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def transformer_genomic(x, y, markers, d_model: int = 8, lam: float = 1.0,
                        seed: int = 0, deterministic_seed: int | None = None):
    """Single-head self-attention with random key/query/value projections,
    followed by mean-pooling and ridge regression.

    For each individual i with marker sequence M_i ∈ R^{L×1}:
        E = M_i[:, None] @ W_emb        # (L, d_model)  -- random embedding
        Q = E @ W_Q,  K = E @ W_K,  V = E @ W_V
        A = softmax(Q K^T / sqrt(d_model))
        C = A @ V                       # (L, d_model)
        c_i = mean(C, axis=0)           # (d_model,) context
    y_hat = c @ beta + intercept, beta = ridge regression on stacked c's.

    Parameters
    ----------
    x : array-like (n,) or (n,q). Optional fixed-effect features (concatenated).
    y : array-like (n,)
    markers : array-like (n, L)
    d_model : int, default 8.
    lam : float, default 1.0. Ridge for the linear head.
    seed : int
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged.

    Returns
    -------
    RichResult with payload keys estimate, y_hat, beta, attention, n, method.

    References
    ----------
    Vaswani et al. (2017). Attention is all you need.
    Montesinos Lopez et al. (2022), Ch. 15.
    """
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("trfge", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × L)")
    L = M.shape[1]
    M_mean = M.mean(axis=0); M_sd = M.std(axis=0)
    M_sd = np.where(M_sd > 0, M_sd, 1.0)
    Ms = (M - M_mean) / M_sd
    # Random fixed projections (no training, gives reproducible features)
    scale = 1.0 / np.sqrt(d_model)
    W_emb = rng.normal(0, 1.0, size=(1, d_model)) * scale
    W_Q = rng.normal(0, 1.0, size=(d_model, d_model)) * scale
    W_K = rng.normal(0, 1.0, size=(d_model, d_model)) * scale
    W_V = rng.normal(0, 1.0, size=(d_model, d_model)) * scale
    # Positional encoding (sinusoidal)
    pos = np.arange(L)[:, None]
    dim = np.arange(d_model)[None, :]
    div = np.power(10000.0, (2.0 * (dim // 2)) / d_model)
    pe = np.zeros((L, d_model))
    pe[:, 0::2] = np.sin(pos / div[:, 0::2])
    pe[:, 1::2] = np.cos(pos / div[:, 1::2])
    # Compute context vector per individual
    context = np.zeros((n, d_model))
    attentions = np.zeros((n, L, L))
    for i in range(n):
        E = Ms[i].reshape(L, 1) @ W_emb + pe   # (L, d)
        Q = E @ W_Q; K = E @ W_K; V = E @ W_V
        scores = Q @ K.T / np.sqrt(d_model)
        A = _softmax(scores, axis=-1)
        attentions[i] = A
        C = A @ V
        context[i] = C.mean(axis=0)
    # Optional fixed features
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1 and Xa.size > 0:
        Xa = Xa.reshape(-1, 1)
    if Xa.size == 0:
        Xa = np.zeros((n, 0))
    feats = np.column_stack([np.ones(n), context, Xa])
    A_mat = feats.T @ feats + lam * np.eye(feats.shape[1])
    A_mat[0, 0] -= lam  # don't regularise intercept
    beta = np.linalg.solve(A_mat, feats.T @ y)
    y_hat = feats @ beta
    resid = y - y_hat
    se = float(np.sqrt(np.mean(resid ** 2)))
    return RichResult(
        title="Transformer (1-head self-attention) genomic predictor",
        summary_lines=[
            ("n", n),
            ("L (sequence length)", L),
            ("d_model", d_model),
            ("ridge lambda", lam),
            ("residual SE", se),
        ],
        payload={
            "estimate": float(np.mean(y_hat)),
            "y_hat": y_hat,
            "beta": beta,
            "attention": attentions,
            "context": context,
            "se": se,
            "n": n,
            "method": "Transformer 1-head random-projection + ridge head",
        },
        warnings=["Random fixed projections (no attention training): "
                  "captures positional+linear structure; for production "
                  "trainable transformers use torch/keras."],
    )


def cheatsheet():
    return "trfge: Transformer attention genomic predictor"


# CANONICAL TEST
# np.random.seed(9); M = np.random.randn(12, 6); y = M[:,2] + 0.2*np.random.randn(12)
# r = transformer_genomic(np.zeros(12), y, M, seed=9); r.attention.shape==(12,6,6).
