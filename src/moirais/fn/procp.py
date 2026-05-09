# moirais.fn — function file (hadesllm/moirais)
"""Procrustes rotation for factor analysis or PCA loadings.

Finds optimal orthogonal or oblique transformation to match a target matrix.
Useful for aligning factor loadings across studies or rotations.

References
----------
Gower, J. C., & Dijksterhuis, G. B. (2004). Procrustes Problems. Oxford University Press.
"""

import numpy as np
from scipy import linalg as la

__all__ = ["procp"]


def procp(X, Y, orthogonal=True, scale=False):
    """
    Procrustes rotation: find optimal transformation of X to match Y.

    Parameters
    ----------
    X : ndarray, shape (p, k)
        Matrix to be transformed (e.g., observed loadings).
    Y : ndarray, shape (p, k)
        Target matrix (e.g., hypothesized loadings).
    orthogonal : bool, optional
        If True (default), find orthogonal transformation; else unrestricted.
    scale : bool, optional
        If True, allow scaling; else shape matching only (default False).

    Returns
    -------
    dict
        'X_transformed' : ndarray, shape (p, k)
            Rotated version of X.
        'R' : ndarray
            Rotation/transformation matrix.
        'scale_factor' : float
            Scaling factor (if scale=True).
        'residual' : float
            Sum of squared residuals after rotation.
        'disparity' : float
            Proportion of X variance unexplained by rotation.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    p, k = X.shape
    assert Y.shape == (p, k), "X and Y must have same shape"

    # Centering
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # SVD of X^T Y
    U, _, Vt = la.svd(X_centered.T @ Y_centered)
    R = U @ Vt

    # Transform
    if not orthogonal:
        # Unrestricted least squares
        R = la.lstsq(X_centered, Y_centered)[0]

    X_transformed = X_centered @ R

    # Scaling (optional)
    scale_factor = 1.0
    if scale:
        scale_factor = np.sum(X_transformed * Y_centered) / np.sum(X_transformed**2)
        X_transformed = scale_factor * X_transformed

    # Residuals
    residual = np.sum((X_transformed - Y_centered) ** 2)
    var_X = np.sum(X_centered**2)
    disparity = residual / var_X if var_X > 0 else 0

    return {
        "X_transformed": X_transformed + Y.mean(axis=0),
        "R": R,
        "scale_factor": scale_factor,
        "residual": residual,
        "disparity": disparity,
    }
