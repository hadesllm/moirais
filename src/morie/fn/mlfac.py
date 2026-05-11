# morie.fn — function file (hadesllm/morie)
"""Maximum Likelihood Factor Analysis.

Estimates factor analysis via maximum likelihood, optimizing the likelihood
subject to the factor model. Returns loadings, communalities, and fit statistics.

References
----------
Jöreskog, K. G. (1967). Some contributions to maximum likelihood factor analysis.
    Psychometrika, 32(4), 443-482.
Anderson, T. W., & Rubin, H. (1956). Statistical inference in factor analysis.
    In Proc. 3rd Berkeley Symp. Mathematical Statistics and Probability.
"""

import numpy as np
from scipy import linalg as la

__all__ = ["mlfac"]


def mlfac(
    X,
    n_factors=None,
    max_iter=200,
    tol=1e-6,
    scale=True,
):
    """
    Maximum Likelihood Factor Analysis.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix with n observations and p variables.
    n_factors : int, optional
        Number of factors. If None, uses Kaiser criterion (eigenvalues > 1).
    max_iter : int, optional
        Maximum iterations for EM algorithm (default 200).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    scale : bool, optional
        If True (default), use correlation matrix; else covariance.

    Returns
    -------
    dict
        'loadings' : ndarray, shape (p, n_factors)
            Maximum likelihood factor loadings.
        'communalities' : ndarray, shape (p,)
            Communalities.
        'uniqueness' : ndarray, shape (p,)
            Unique variances (1 - communality).
        'variance_explained' : ndarray, shape (n_factors,)
            Variance explained by each factor.
        'log_likelihood' : float
            Log-likelihood of the model.
        'aic' : float
            Akaike Information Criterion.
        'bic' : float
            Bayesian Information Criterion.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # Standardize
    X_centered = X - X.mean(axis=0)
    if scale:
        std = X_centered.std(axis=0, ddof=1)
        std[std == 0] = 1
        X_scaled = X_centered / std
        S = np.cov(X_scaled, rowvar=False)
    else:
        S = np.cov(X_centered, rowvar=False)

    # Determine number of factors
    if n_factors is None:
        eigenvalues, _ = la.eigh(S)
        eigenvalues = np.sort(eigenvalues)[::-1]
        n_factors = np.sum(eigenvalues > 1)
        if n_factors == 0:
            n_factors = 1

    # Initialize with PCA loadings
    eigenvalues, eigenvectors = la.eigh(S)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx[:n_factors]]

    L = eigenvectors * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))
    Psi = np.diag(np.maximum(1 - np.sum(L**2, axis=1), 0.01))

    # EM algorithm
    for iteration in range(max_iter):
        L_old = L.copy()

        # E-step: compute factor scores covariance
        # Cov(F) = (I + L^T Psi^{-1} L)^{-1}
        Psi_inv = np.diag(1 / np.diag(Psi))
        M = np.eye(n_factors) + L.T @ Psi_inv @ L
        M_inv = la.inv(M)

        # M-step: update loadings and uniqueness
        L_new = S @ Psi_inv @ L @ M_inv
        Psi_new = np.diag(np.diag(S) - np.diag(L_new @ M_inv @ L_new.T))
        Psi_new = np.diag(np.maximum(np.diag(Psi_new), 0.01))

        # Check convergence
        if np.max(np.abs(L_new - L_old)) < tol:
            L = L_new
            Psi = Psi_new
            break

        L = L_new
        Psi = Psi_new

    # Communalities and uniqueness
    h2 = np.sum(L**2, axis=1)
    psi = np.diag(Psi)

    # Log-likelihood
    # LL = -0.5 * n * (p * log(2*pi) + log|Sigma| + tr(S @ Sigma^{-1}))
    # where Sigma = L @ L^T + Psi
    Sigma = L @ L.T + Psi
    try:
        log_det = np.linalg.slogdet(Sigma)[1]
        Sigma_inv = la.inv(Sigma)
        trace_term = np.trace(S @ Sigma_inv)
        log_likelihood = -0.5 * n * (p * np.log(2 * np.pi) + log_det + trace_term)
    except la.LinAlgError:
        log_likelihood = np.nan

    # Information criteria
    n_params = p * n_factors + p  # loadings + uniqueness variances
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n)

    variance_explained = np.sum(L**2, axis=0)

    return {
        "loadings": L,
        "communalities": h2,
        "uniqueness": 1 - h2,
        "variance_explained": variance_explained,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
    }
