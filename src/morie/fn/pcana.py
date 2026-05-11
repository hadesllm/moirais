# morie.fn — function file (hadesllm/morie)
"""Principal Component Analysis (PCA) via eigendecomposition.

Performs PCA on a data matrix X via spectral decomposition of the correlation
or covariance matrix. Returns principal components, loadings, and explained variance.

References
----------
Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.
Jackson, J. E. (1991). A User's Guide to Principal Components. Wiley.
"""

import numpy as np
from scipy import linalg as la

__all__ = ["pcana"]


def pcana(
    X,
    n_components=None,
    scale=True,
    return_loadings=True,
):
    """
    Principal Component Analysis via eigendecomposition.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix with n observations and p variables.
    n_components : int, optional
        Number of components to return. If None, return all.
    scale : bool, optional
        If True (default), use correlation matrix; else covariance.
    return_loadings : bool, optional
        If True (default), return loadings (correlations of variables with PCs).

    Returns
    -------
    dict
        'components' : ndarray, shape (n_components, p)
            Principal component coefficients (eigenvectors).
        'scores' : ndarray, shape (n, n_components)
            Scores (projections) of observations onto PCs.
        'explained_var' : ndarray, shape (n_components,)
            Variance explained by each PC.
        'cum_var_explained' : ndarray, shape (n_components,)
            Cumulative variance explained.
        'loadings' : ndarray, shape (n_components, p), optional
            Correlations of variables with PCs (if return_loadings=True).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if n_components is None:
        n_components = min(n, p)

    # Center data
    X_centered = X - X.mean(axis=0)

    # Compute covariance or correlation matrix
    if scale:
        # Correlation: divide by std
        std = X_centered.std(axis=0, ddof=1)
        std[std == 0] = 1
        X_scaled = X_centered / std
        S = np.cov(X_scaled, rowvar=False)
    else:
        S = np.cov(X_centered, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = la.eigh(S)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Retain n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    # Scores: project centered data onto eigenvectors
    if scale:
        scores = X_scaled @ eigenvectors
    else:
        scores = X_centered @ eigenvectors

    # Explained variance (as proportions)
    explained_var = eigenvalues / eigenvalues.sum()
    cum_var_explained = np.cumsum(explained_var)

    result = {
        "components": eigenvectors.T,
        "scores": scores,
        "explained_var": explained_var,
        "cum_var_explained": cum_var_explained,
    }

    if return_loadings:
        loadings = (eigenvectors * np.sqrt(eigenvalues)).T
        result["loadings"] = loadings

    return result
