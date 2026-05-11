# morie.fn — function file (hadesllm/morie)
"""PCA via SVD (pure numpy)."""

import numpy as np

from ._containers import DescriptiveResult


def pca_simple(X: np.ndarray, n_components: int = 2) -> DescriptiveResult:
    """
    Principal Component Analysis via SVD (pure numpy).

    :param X: (n, p) data matrix.
    :param n_components: Number of components to retain.
    :return: DescriptiveResult with scores, loadings, variance ratios.

    References
    ----------
    Jolliffe IT (2002). Principal Component Analysis. 2nd ed. Springer.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eig = s**2 / (n - 1)
    total_var = eig.sum()
    k = min(n_components, len(eig))
    scores = U[:, :k] * s[:k]
    loadings = Vt[:k].T
    var_ratio = eig[:k] / total_var
    return DescriptiveResult(
        name="pca_simple",
        value=float(var_ratio.sum()),
        extra={
            "scores": scores,
            "loadings": loadings,
            "eigenvalues": eig[:k],
            "variance_ratio": var_ratio,
            "n_components": k,
            "n": n,
            "p": p,
        },
    )


pca_ = pca_simple


def cheatsheet() -> str:
    return "pca_simple({}) -> PCA via SVD (pure numpy)."
