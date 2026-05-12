# morie.fn — function file (hadesllm/morie)
"""Dense layer forward pass."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["forward_pass_dense"]


def _apply_activation(z: np.ndarray, activation: str) -> np.ndarray:
    a = activation.lower()
    if a in ("identity", "linear", "none"):
        return z
    if a == "sigmoid":
        # Numerically stable sigmoid
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out
    if a == "tanh":
        return np.tanh(z)
    if a == "relu":
        return np.maximum(0.0, z)
    if a == "softmax":
        zs = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(zs)
        return e / np.sum(e, axis=-1, keepdims=True)
    raise ValueError(f"Unknown activation: {activation!r}")


def forward_pass_dense(x, w, b, activation: str = "sigmoid"):
    r"""Dense layer forward pass.

    Computes :math:`z = W x + b` followed by an element-wise
    activation :math:`a = \\sigma(z)`.

    Parameters
    ----------
    x : array-like, shape ``(n_in,)`` or ``(batch, n_in)``
        Input data (single sample or mini-batch).
    w : array-like, shape ``(n_out, n_in)``
        Weight matrix.
    b : array-like, shape ``(n_out,)``
        Bias vector.
    activation : str
        ``'sigmoid'`` (default), ``'tanh'``, ``'relu'``, ``'softmax'``, or
        ``'identity'``.

    Returns
    -------
    result : RichResult
        Keys: ``z`` (pre-activation), ``a`` / ``estimate`` (post-activation).

    References
    ----------
    Goodfellow, Bengio & Courville (2016) *Deep Learning*, Ch 6.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    b = np.asarray(b, dtype=float)
    # x @ W^T + b  (broadcasts over batch dim if present)
    z = x @ w.T + b
    a = _apply_activation(z, activation)
    return RichResult(
        title="Dense layer forward pass",
        summary_lines=[("Activation", activation), ("Output shape", a.shape)],
        payload={
            "z": z,
            "a": a,
            "estimate": a,
            "activation": activation,
            "method": "Dense layer forward pass",
        },
    )


# CANONICAL TEST
# x=[1,2], W=[[1,0],[0,1],[1,1]], b=[0,0,0], identity -> z=a=[1,2,3]
# softmax of [1,2,3] -> [0.0900, 0.2447, 0.6652]


def cheatsheet():
    return "fwpas: Dense layer forward pass z=Wx+b, a=sigma(z)"
