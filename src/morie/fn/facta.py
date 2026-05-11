# morie.fn — function file (hadesllm/morie)
"""Factor Analysis via principal axis factoring.

Performs factor analysis using principal axis factoring (PAF) with iterative
communality estimation. Returns factor loadings, communalities, and uniqueness.

References
----------
Harman, H. H. (1976). Modern Factor Analysis (3rd ed.). University of Chicago Press.
Fabrigar, L. R., et al. (1999). Evaluating the use of exploratory factor analysis
    in psychological research. Psychological Methods, 4(3), 272-299.
"""

import numpy as np
from scipy import linalg as la

__all__ = ["facta"]


def facta(
    X,
    n_factors=None,
    max_iter=100,
    tol=1e-6,
    scale=True,
):
    """
    Factor Analysis via Principal Axis Factoring.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix with n observations and p variables.
    n_factors : int, optional
        Number of factors to extract. If None, uses Kaiser criterion (eigenvalues > 1).
    max_iter : int, optional
        Maximum iterations for communality estimation (default 100).
    tol : float, optional
        Convergence tolerance for communalities (default 1e-6).
    scale : bool, optional
        If True (default), use correlation matrix; else covariance.

    Returns
    -------
    dict
        'loadings' : ndarray, shape (p, n_factors)
            Factor loadings (correlations of variables with factors).
        'communalities' : ndarray, shape (p,)
            Communalities (variance explained by factors for each variable).
        'uniqueness' : ndarray, shape (p,)
            Uniqueness (1 - communality).
        'variance_explained' : ndarray, shape (n_factors,)
            Variance explained by each factor.
        'iterations' : int
            Number of iterations to convergence.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # Center and standardize
    X_centered = X - X.mean(axis=0)
    if scale:
        std = X_centered.std(axis=0, ddof=1)
        std[std == 0] = 1
        X_scaled = X_centered / std
        R = np.cov(X_scaled, rowvar=False)
    else:
        R = np.cov(X_centered, rowvar=False)

    # Initialize communalities with squared multiple correlation
    # Start with diagonal of inv(R)^-1
    try:
        R_inv = la.inv(R)
        h2 = 1 - 1 / np.diag(R_inv)
        h2 = np.clip(h2, 0.01, 0.99)
    except la.LinAlgError:
        h2 = 0.5 * np.ones(p)

    # Iterative communality estimation
    for iteration in range(max_iter):
        h2_old = h2.copy()

        # Reduced correlation matrix
        R_reduced = R.copy()
        np.fill_diagonal(R_reduced, h2)

        # Eigendecomposition
        eigenvalues, eigenvectors = la.eigh(R_reduced)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine number of factors
        if n_factors is None:
            n_factors_use = np.sum(eigenvalues > 1)
            if n_factors_use == 0:
                n_factors_use = 1
        else:
            n_factors_use = min(n_factors, len(eigenvalues))

        # Keep positive eigenvalues only
        valid = eigenvalues > 0
        eigenvalues = eigenvalues[valid]
        eigenvectors = eigenvectors[:, valid]

        if n_factors_use > len(eigenvalues):
            n_factors_use = len(eigenvalues)

        eigenvalues = eigenvalues[:n_factors_use]
        eigenvectors = eigenvectors[:, :n_factors_use]

        # Loadings
        loadings = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))

        # Update communalities
        h2 = np.sum(loadings**2, axis=1)
        h2 = np.clip(h2, 0, 0.99)

        # Check convergence
        if np.max(np.abs(h2 - h2_old)) < tol:
            break

    # Final results
    variance_explained = np.sum(loadings**2, axis=0)
    uniqueness = 1 - h2

    return {
        "loadings": loadings,
        "communalities": h2,
        "uniqueness": uniqueness,
        "variance_explained": variance_explained,
        "iterations": iteration + 1,
    }
