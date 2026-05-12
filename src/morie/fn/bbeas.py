# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Shape analysis (morphometrics). 'Dude, I can turn into a T-Rex.' -- Beast Boy"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def procrustes_shape(
    X: np.ndarray,
    Y: np.ndarray,
) -> DescriptiveResult:
    """Ordinary Procrustes analysis: align shape Y to reference shape X.

    Finds the optimal rotation, scaling, and translation that minimizes
    the sum of squared distances between corresponding landmarks.

    Parameters
    ----------
    X : array (k, p)
        Reference landmark coordinates (k landmarks in p dimensions).
    Y : array (k, p)
        Target landmark coordinates (same shape as X).

    Returns
    -------
    DescriptiveResult
        ``value`` = Procrustes distance (disparity) after alignment.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X={X.shape}, Y={Y.shape}")
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("Need at least 3 landmarks in 2-D matrix")
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    sx = np.sqrt(np.sum(X_c**2))
    sy = np.sqrt(np.sum(Y_c**2))
    if sx < 1e-10 or sy < 1e-10:
        raise ValueError("Degenerate configuration (zero size)")
    X_n = X_c / sx
    Y_n = Y_c / sy
    U, S, Vt = np.linalg.svd(X_n.T @ Y_n)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Y_aligned = Y_n @ R
    disparity = float(np.sum((X_n - Y_aligned) ** 2))
    scale = float(np.sum(S)) * sx / sy
    return DescriptiveResult(
        name="Procrustes shape analysis",
        value=float(np.sqrt(disparity)),
        extra={
            "disparity": disparity,
            "scale": scale,
            "rotation_det": float(np.linalg.det(R)),
            "n_landmarks": X.shape[0],
            "n_dims": X.shape[1],
            "aligned": Y_aligned.tolist(),
        },
    )


bbeas = procrustes_shape


def cheatsheet() -> str:
    return "procrustes_shape({}) -> Shape analysis (morphometrics). 'Dude, I can turn into a T-R"
