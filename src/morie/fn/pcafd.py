# morie.fn — function file (hadesllm/morie)
"""PCA feature extraction."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luck is what happens when preparation meets opportunity. — Seneca"


def pca_features(X, n_components=None, **kwargs) -> DescriptiveResult:
    """
    Principal Component Analysis via eigendecomposition of the
    covariance matrix.

    :param X: (n, d) data matrix.
    :param n_components: Number of components to keep. Default: all.
    :return: DescriptiveResult with transformed data and explained variance.

    References
    ----------
    Pearson K (1901). On lines and planes of closest fit to
    systems of points in space. Philosophical Magazine, 2(11),
    559-572.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if n_components is None:
        n_components = d
    n_components = min(n_components, d)
    W = eigvecs[:, :n_components]
    X_proj = X_centered @ W
    total_var = eigvals.sum()
    explained = eigvals[:n_components] / total_var if total_var > 0 else eigvals[:n_components]
    return DescriptiveResult(
        name="pca_features",
        value=float(np.sum(explained)),
        extra={
            "n_components": n_components,
            "explained_variance_ratio": explained.tolist(),
            "eigenvalues": eigvals.tolist(),
            "transformed_shape": list(X_proj.shape),
        },
    )


pcafd = pca_features


def cheatsheet() -> str:
    return "pca_features({}) -> PCA feature extraction."
