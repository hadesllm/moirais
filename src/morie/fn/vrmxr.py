"""Varimax rotation of factor loadings.

Orthogonal rotation that maximizes variance of squared loadings, resulting in
a "simple structure" with each variable loading highly on few factors.

References
----------
Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis.
    Psychometrika, 23(3), 187-200.
"""

import numpy as np

__all__ = ["vrmxr"]


def vrmxr(L, max_iter=1000, tol=1e-6, gamma=1.0):
    """
    Varimax rotation of factor loadings.

    Parameters
    ----------
    L : ndarray, shape (p, k)
        Factor loadings matrix.
    max_iter : int, optional
        Maximum iterations (default 1000).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    gamma : float, optional
        Rotation parameter (default 1.0 for standard varimax).

    Returns
    -------
    dict
        'loadings' : ndarray, shape (p, k)
            Rotated loadings.
        'R' : ndarray, shape (k, k)
            Rotation matrix.
        'variance' : float
            Sum of variances of squared loadings (objective).
        'iterations' : int
            Iterations to convergence.
    """
    L = np.asarray(L, dtype=float)
    p, k = L.shape

    # Normalize
    L_norm = np.sqrt(np.sum(L**2, axis=1, keepdims=True))
    L_norm[L_norm == 0] = 1
    L_normalized = L / L_norm

    R = np.eye(k)

    for iteration in range(max_iter):
        R_old = R.copy()

        # For each pair of columns
        for i in range(k):
            for j in range(i + 1, k):
                # Extract columns
                a = L_normalized[:, i]
                b = L_normalized[:, j]

                # Compute rotation angle to maximize variance
                # Using the Kaiser criterion
                u = a**2 - b**2
                v = 2 * a * b
                numerator = np.sum(v * (u))
                denominator = np.sum(u**2 - v**2)

                if abs(denominator) < 1e-10:
                    theta = 0
                else:
                    theta = 0.25 * np.arctan2(numerator, denominator)

                # Apply rotation to columns i, j
                cos_t = np.cos(gamma * theta)
                sin_t = np.sin(gamma * theta)

                a_new = cos_t * a + sin_t * b
                b_new = -sin_t * a + cos_t * b

                L_normalized[:, i] = a_new
                L_normalized[:, j] = b_new

                # Update R
                Rij = np.eye(k)
                Rij[i, i] = cos_t
                Rij[i, j] = sin_t
                Rij[j, i] = -sin_t
                Rij[j, j] = cos_t
                R = R @ Rij

        # Check convergence
        if np.max(np.abs(R - R_old)) < tol:
            break

    # Restore original scale
    L_rotated = L_normalized * L_norm

    # Compute objective: sum of variances of squared loadings per column
    variance = np.sum(np.var(L_rotated**2, axis=0))

    return {
        "loadings": L_rotated,
        "R": R,
        "variance": variance,
        "iterations": iteration + 1,
    }
