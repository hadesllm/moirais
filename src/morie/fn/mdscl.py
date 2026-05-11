# morie.fn — function file (hadesllm/morie)
"""Classical Multidimensional Scaling (MDS).

Performs classical MDS via eigendecomposition of the double-centered distance matrix.
Projects high-dimensional data into lower dimensions while preserving pairwise distances.

References
----------
Torgerson, W. S. (1958). Theory and Methods of Scaling. Wiley.
Cox, T. F., & Cox, M. A. (2001). Multidimensional Scaling (2nd ed.). Chapman & Hall.
"""

import numpy as np
from scipy import linalg as la

__all__ = ["mdscl"]


def mdscl(D, n_dims=2):
    """
    Classical Multidimensional Scaling.

    Parameters
    ----------
    D : ndarray, shape (n, n)
        Symmetric distance or dissimilarity matrix.
    n_dims : int, optional
        Number of dimensions for the output (default 2).

    Returns
    -------
    dict
        'coordinates' : ndarray, shape (n, n_dims)
            Embedded coordinates in the lower-dimensional space.
        'stress' : float
            Goodness-of-fit (sum of squared residuals).
        'eigenvalues' : ndarray, shape (n_dims,)
            Eigenvalues corresponding to the dimensions.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    assert D.shape == (n, n), "D must be square"

    # Square distances
    D_sq = D**2

    # Double centering: J = I - 1/n * 1^T
    J = np.eye(n) - np.ones((n, n)) / n

    # B = -0.5 * J @ D_sq @ J
    B = -0.5 * J @ D_sq @ J

    # Eigendecomposition
    eigenvalues, eigenvectors = la.eigh(B)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalues and requested dimensions
    valid = eigenvalues > 1e-10
    eigenvalues_valid = eigenvalues[valid]
    eigenvectors_valid = eigenvectors[:, valid]

    n_keep = min(n_dims, len(eigenvalues_valid))
    eigenvalues_keep = eigenvalues_valid[:n_keep]
    eigenvectors_keep = eigenvectors_valid[:, :n_keep]

    # Coordinates: X = V * sqrt(Lambda)
    X = eigenvectors_keep * np.sqrt(np.maximum(eigenvalues_keep, 0))

    # Reconstruct distances
    distances_reconstructed = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances_reconstructed[i, j] = np.linalg.norm(X[i] - X[j])

    # Stress (sum of squared residuals)
    stress = np.sum((D - distances_reconstructed) ** 2)

    return {
        "coordinates": X,
        "stress": stress,
        "eigenvalues": eigenvalues_keep,
    }
