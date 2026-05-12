# morie.fn -- function file (hadesllm/morie)
"""Root-mean-square normalisation (Zhang & Sennrich 2019)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rms_norm"]


def rms_norm(x, gamma=None, eps: float = 1e-6):
    """RMSNorm:  ``y_i = x_i / sqrt(mean(x^2) + eps) * gamma_i``.

    Normalises along the last axis (the feature/embedding dimension)
    and rescales by an optional learnable diagonal weight ``gamma``.

    Parameters
    ----------
    x : array-like, shape (..., d)
    gamma : array-like, shape (d,), optional
        Per-feature scale (default = 1).
    eps : float
        Numerical-stability epsilon.

    Returns
    -------
    RichResult with keys: tensor (y), rms.
    """
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    y = x / rms
    if gamma is not None:
        gamma = np.asarray(gamma, dtype=float)
        y = y * gamma
    return RichResult(
        title="RMSNorm (Zhang & Sennrich 2019)",
        summary_lines=[("shape", y.shape),
                       ("rms_mean", float(np.mean(rms)))],
        payload={"tensor": y, "rms": np.squeeze(rms, axis=-1),
                 "eps": eps, "method": "RMSNorm"},
    )


def cheatsheet():
    return "rmsnr(x, gamma, eps): RMS normalisation along last axis"


# CANONICAL TEST
# >>> x = np.array([[3.0, 4.0]])
# >>> r = rms_norm(x, eps=0.0)
# >>> np.allclose(r["tensor"], x / np.sqrt(12.5))
# True
