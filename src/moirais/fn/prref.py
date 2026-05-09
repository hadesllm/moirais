# moirais.fn — function file (hadesllm/moirais)
"""Procrustes reflection and rotation. 'Cleave.' -- Sukuna, Jujutsu Kaisen"""

from __future__ import annotations

from ._containers import DescriptiveResult


def procrustes_reflection(X, Y):
    """Align X to Y via Procrustes (rotation + reflection).

    Parameters
    ----------
    X : array-like
        Source coordinate matrix (n x p).
    Y : array-like
        Target coordinate matrix (n x p).

    Returns
    -------
    DescriptiveResult
        value = aligned X, extra has rotation matrix and disparity.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    M = Y_c.T @ X_c
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    X_aligned = X_c @ R.T
    disparity = float(np.sum((X_aligned - Y_c) ** 2))
    return DescriptiveResult(
        name="procrustes_reflection",
        value=X_aligned,
        extra={"rotation": R, "disparity": disparity},
    )


prref = procrustes_reflection


def cheatsheet() -> str:
    return "procrustes_reflection({}) -> Procrustes reflection and rotation. 'Cleave.' -- Sukuna, Juj"
