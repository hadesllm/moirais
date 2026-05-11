# morie.fn — function file (hadesllm/morie)
"""Randomized PCA. 'This is the way.' -- Din Djarin"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def fast_pca(
    X: np.ndarray,
    n_components: int | None = None,
    seed: int = 42,
) -> DescriptiveResult:
    """Randomized PCA via the Halko-Martinsson-Tropp algorithm.

    Uses a randomized SVD to compute the top principal components
    efficiently, especially for large matrices.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data (centered internally).
    n_components : int or None
        Number of components to retain. If None, uses min(n, p).
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        name='Fast PCA', value=explained variance ratio (total),
        extra has 'components' (loadings), 'scores' (projected data),
        'explained_variance', 'explained_variance_ratio',
        'singular_values'.

    References
    ----------
    Halko, N., Martinsson, P.G. & Tropp, J.A. (2011). Finding
    structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions. *SIAM Review*,
    53(2), 217-288. doi:10.1137/090771806
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    mean = X.mean(axis=0)
    X_centered = X - mean

    k = n_components if n_components is not None else min(n, p)
    k = min(k, min(n, p))

    oversampling = min(10, min(n, p) - k)
    l = k + oversampling

    Omega = rng.standard_normal((p, l))
    Y = X_centered @ Omega

    Q, _ = np.linalg.qr(Y)

    for _ in range(2):
        Z = X_centered.T @ Q
        Q, _ = np.linalg.qr(X_centered @ Z)

    B = Q.T @ X_centered
    U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)

    s = s[:k]
    Vt = Vt[:k]
    U_hat = U_hat[:, :k]

    scores = Q @ U_hat * s[None, :]

    total_var = np.sum(X_centered**2) / (n - 1)
    explained_var = s**2 / (n - 1)
    explained_ratio = explained_var / total_var if total_var > 0 else explained_var

    return DescriptiveResult(
        name="Fast PCA",
        value=float(np.sum(explained_ratio)),
        extra={
            "components": Vt,
            "scores": scores,
            "explained_variance": explained_var,
            "explained_variance_ratio": explained_ratio,
            "singular_values": s,
            "mean": mean,
            "n_components": k,
        },
    )


def cheatsheet() -> str:
    return "fast_pca({}) -> Randomized PCA. 'This is the way.' -- Din Djarin"
