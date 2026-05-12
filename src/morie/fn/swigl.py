# morie.fn — function file (hadesllm/morie)
"""SwiGLU gated activation (Shazeer 2020)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["swiglu_activation"]


def _silu(z):
    # Swish/SiLU = z * sigmoid(z); numerically stable form.
    return z * (1.0 / (1.0 + np.exp(-z)))


def swiglu_activation(x, W=None, V=None, b=None, c=None):
    """SwiGLU: ``Swish(xW + b) * (xV + c)``.

    Parameters
    ----------
    x : ndarray, shape (..., d_in)
    W, V : ndarray, shape (d_in, d_out), optional
        Linear projection matrices.  If both None, defaults to
        identity (element-wise gated activation).
    b, c : ndarray, shape (d_out,), optional
        Biases (default zero).

    Returns
    -------
    RichResult with keys: tensor (output), gate, up.
    """
    x = np.asarray(x, dtype=float)
    if W is None and V is None:
        d_out = x.shape[-1]
        W = np.eye(d_out)
        V = np.eye(d_out)
    elif W is None or V is None:
        raise ValueError("Provide both W and V or neither")
    W = np.asarray(W, dtype=float)
    V = np.asarray(V, dtype=float)
    b = np.zeros(W.shape[1]) if b is None else np.asarray(b, dtype=float)
    c = np.zeros(V.shape[1]) if c is None else np.asarray(c, dtype=float)
    gate = _silu(x @ W + b)
    up = x @ V + c
    out = gate * up
    return RichResult(
        title="SwiGLU Activation (Shazeer 2020)",
        summary_lines=[("shape", out.shape),
                       ("mean", float(np.mean(out)))],
        payload={"tensor": out, "gate": gate, "up": up,
                 "method": "SwiGLU"},
    )


def cheatsheet():
    return "swigl(x, W, V, b, c): Swish(xW+b) * (xV+c)"


# CANONICAL TEST
# >>> x = np.array([[1.0, 0.0]]); W = V = np.eye(2)
# >>> r = swiglu_activation(x, W=W, V=V)
# >>> r["tensor"].shape
# (1, 2)
