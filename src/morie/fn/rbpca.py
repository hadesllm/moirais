# morie.fn — function file (hadesllm/morie)
"""Robust PCA via projection pursuit."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def robust_pca_pp(
    X,
    *,
    n_components: int = 2,
    n_directions: int = 250,
    seed: int = 42,
) -> ESRes:
    """Robust PCA via projection pursuit.

    Finds directions that maximise a robust dispersion measure
    (MAD) rather than classical variance.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Centred data matrix.
    n_components : int
        Number of components to extract.
    n_directions : int
        Number of random directions to search per component.
    seed : int
        RNG seed.

    Returns
    -------
    ESRes
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    n_components = min(n_components, p)
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    rng = np.random.default_rng(seed)
    R = X - np.median(X, axis=0)
    loadings = np.zeros((p, n_components))
    variances = np.zeros(n_components)

    for k in range(n_components):
        best_mad = -1.0
        best_dir = np.zeros(p)
        for _ in range(n_directions):
            d = rng.standard_normal(p)
            d /= np.linalg.norm(d) + 1e-12
            proj = R @ d
            mad_val = 1.4826 * np.median(np.abs(proj - np.median(proj)))
            if mad_val > best_mad:
                best_mad = mad_val
                best_dir = d
        loadings[:, k] = best_dir
        variances[k] = best_mad**2
        proj = (R @ best_dir).reshape(-1, 1)
        R = R - proj @ best_dir.reshape(1, -1)

    return ESRes(
        measure="robust_pca_pp",
        estimate=float(np.sum(variances)),
        n=n,
        extra={
            "loadings": loadings.tolist(),
            "robust_variances": variances.tolist(),
            "n_components": n_components,
        },
    )


rbpca = robust_pca_pp


def cheatsheet() -> str:
    return "robust_pca_pp(X) -> Robust PCA via projection pursuit."
