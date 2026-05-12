# morie.fn — function file (hadesllm/morie)
"""Batch normalization forward pass."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["batch_norm_forward"]


def batch_norm_forward(x, gamma=None, beta=None, eps: float = 1e-5, axis: int = 0):
    """Batch normalization forward pass.

    .. math::

        \\mu = \\tfrac{1}{m}\\sum_i x_i, \\qquad
        \\sigma^2 = \\tfrac{1}{m}\\sum_i (x_i - \\mu)^2

        \\hat x_i = \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}, \\qquad
        y_i = \\gamma \\hat x_i + \\beta

    Parameters
    ----------
    x : array-like, shape ``(batch, features)``
        Input mini-batch.
    gamma : array-like, optional
        Per-feature scale. Defaults to ones.
    beta : array-like, optional
        Per-feature shift. Defaults to zeros.
    eps : float
        Numerical-stability constant.
    axis : int
        Axis to normalize over (default 0, the batch axis).

    Returns
    -------
    result : RichResult
        Keys: ``y`` / ``estimate``, ``x_hat``, ``mu``, ``var``.

    References
    ----------
    Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML*.
    """
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    if gamma is None:
        gamma = np.ones(x_hat.shape[-1])
    if beta is None:
        beta = np.zeros(x_hat.shape[-1])
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    y = gamma * x_hat + beta
    return RichResult(
        title="Batch normalization (forward)",
        summary_lines=[("mu mean", float(mu.mean())),
                       ("var mean", float(var.mean()))],
        payload={
            "y": y,
            "estimate": y,
            "x_hat": x_hat,
            "mu": np.squeeze(mu),
            "var": np.squeeze(var),
            "eps": eps,
            "method": "Batch normalization forward",
        },
    )


# CANONICAL TEST
# x = [[1,2,3],[4,5,6]] axis=0 -> mu=[2.5,3.5,4.5], var=[2.25,2.25,2.25]
# x_hat row 0 ~ [-1,-1,-1]; x_hat row 1 ~ [+1,+1,+1] (modulo eps).


def cheatsheet():
    return "bnfwd: BatchNorm y = gamma*(x-mu)/sqrt(var+eps)+beta"
