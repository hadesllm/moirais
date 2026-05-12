# morie.fn — function file (hadesllm/morie)
"""Backpropagation gradient computation for a single dense layer."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["backpropagation"]


def backpropagation(x, y, w=None, b=None, activation: str = "sigmoid"):
    r"""Backpropagation gradient computation for one dense layer with MSE loss.

    For ``L = 1/(2n) * sum( (a - y)^2 )`` with ``a = sigma(Wx + b)``:

    .. math::

        \\frac{\\partial L}{\\partial W} = \\delta\\, x^\\top, \\quad
        \\frac{\\partial L}{\\partial b} = \\delta, \\quad
        \\delta = (a - y) \\odot \\sigma'(z)

    Parameters
    ----------
    x : array-like, shape ``(n_in,)`` or ``(batch, n_in)``
        Inputs.
    y : array-like, shape ``(n_out,)`` or ``(batch, n_out)``
        Targets.
    w : array-like, shape ``(n_out, n_in)``, optional
        Weight matrix. If ``None``, identity-shaped weights are used.
    b : array-like, shape ``(n_out,)``, optional
        Bias vector. If ``None``, zeros are used.
    activation : str
        ``'sigmoid'`` (default), ``'tanh'``, ``'relu'``, ``'identity'``.

    Returns
    -------
    result : RichResult
        Keys: ``dW``, ``db``, ``dx``, ``loss``, ``estimate`` (= ``loss``).

    References
    ----------
    Rumelhart, Hinton & Williams (1986) *Nature* 323, 533-536.
    Goodfellow, Bengio & Courville (2016) *Deep Learning*, Ch 6.5.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n_in = x.shape[1]
    n_out = y.shape[1]
    if w is None:
        w = np.eye(n_out, n_in)
    else:
        w = np.asarray(w, dtype=float)
    if b is None:
        b = np.zeros(n_out)
    else:
        b = np.asarray(b, dtype=float)

    z = x @ w.T + b  # (batch, n_out)
    a = _sigma(z, activation)
    dsig = _sigma_prime(z, activation, a)

    batch = x.shape[0]
    diff = (a - y)  # (batch, n_out)
    loss = float(0.5 * np.sum(diff * diff) / batch)

    delta = diff * dsig  # (batch, n_out)
    dW = delta.T @ x / batch  # (n_out, n_in)
    db = np.sum(delta, axis=0) / batch  # (n_out,)
    dx = delta @ w / batch  # (batch, n_in)

    return RichResult(
        title="Backpropagation (1-layer MSE)",
        summary_lines=[("Loss", loss), ("|dW|", float(np.linalg.norm(dW)))],
        payload={
            "loss": loss,
            "estimate": loss,
            "dW": dW,
            "db": db,
            "dx": dx,
            "a": a,
            "z": z,
            "method": "Backpropagation gradient computation",
        },
    )


def _sigma(z, activation):
    a = activation.lower()
    if a in ("identity", "linear", "none"):
        return z
    if a == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if a == "tanh":
        return np.tanh(z)
    if a == "relu":
        return np.maximum(0.0, z)
    raise ValueError(f"Unknown activation: {activation!r}")


def _sigma_prime(z, activation, a):
    name = activation.lower()
    if name in ("identity", "linear", "none"):
        return np.ones_like(z)
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return (z > 0).astype(z.dtype)
    raise ValueError(f"Unknown activation: {activation!r}")


# CANONICAL TEST
# x=[[1.0]], y=[[0.0]], w=[[1.0]], b=[0.0], sigmoid:
#   z=1.0, a=sigmoid(1)=0.7310586, loss=0.5*0.7310586^2=0.2671501
#   dsig=a*(1-a)=0.1966119, delta=a*dsig=0.1437001
#   dW=0.1437001, db=0.1437001


def cheatsheet():
    return "bkprp: Backpropagation gradients (dW, db, dx) for MSE+sigma"
