# morie.fn — function file (hadesllm/morie)
"""Feature whitening (ZCA)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Difficulties strengthen the mind, as labor does the body. — Seneca"


def feature_whiten(X, method="zca", eps=1e-5, **kwargs) -> DescriptiveResult:
    """Feature whitening via ZCA or PCA transform.

    ZCA whitening:
    :math:`\\mathbf{X}_{\\text{zca}} = \\mathbf{U} \\Lambda^{-1/2} \\mathbf{U}^\\top (\\mathbf{X} - \\mu)`

    PCA whitening: :math:`\\mathbf{X}_{\\text{pca}} = \\Lambda^{-1/2} \\mathbf{U}^\\top (\\mathbf{X} - \\mu)`

    Parameters
    ----------
    X : array-like of shape (n, p)
        Input features.
    method : str
        "zca" or "pca" (default "zca").
    eps : float
        Regularization for eigenvalues (default 1e-5).

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    X_c = X - mu
    cov = X_c.T @ X_c / max(len(X) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)

    if method == "pca":
        X_white = (X_c @ eigvecs) / np.sqrt(eigvals)
    else:
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        X_white = X_c @ W

    cov_white = np.cov(X_white.T) if X_white.shape[0] > 1 else np.eye(X_white.shape[1])

    return DescriptiveResult(
        name="feature_whiten",
        value=float(np.mean(np.diag(cov_white))),
        extra={
            "X_whitened": X_white,
            "mean": mu,
            "eigenvalues": eigvals,
            "eigenvectors": eigvecs,
            "method": method,
        },
    )


fwtng = feature_whiten


def cheatsheet() -> str:
    return "feature_whiten({}) -> Feature whitening (ZCA)."
