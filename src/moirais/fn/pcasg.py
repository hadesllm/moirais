# moirais.fn — function file (hadesllm/moirais)
"""PCA-based signal decomposition.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 16.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['pcasg']

_QUOTE = "The principal components, reveal the truth. --"


def pcasg(
    X: np.ndarray,
    *,
    n_components: int | None = None,
) -> DescriptiveResult:
    """PCA-based multichannel signal decomposition.

    Parameters
    ----------
    X : array-like
        2-D array (n_channels, n_samples) or (n_samples, n_channels).
        If more columns than rows, assumed (n_channels, n_samples).
    n_components : int or None
        Number of principal components (default all).

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        raise ValueError("Need at least 2-D input.")
    if X.shape[0] > X.shape[1]:
        X = X.T
    n_ch, n_samp = X.shape

    if n_components is None:
        n_components = n_ch
    n_components = min(n_components, n_ch)

    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean
    cov = Xc @ Xc.T / (n_samp - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    W = eigvecs[:, :n_components]
    components = W.T @ Xc
    explained = eigvals[:n_components] / np.sum(eigvals)

    return DescriptiveResult(
        name="pcasg",
        value=float(np.sum(explained)),
        extra={
            "components": components,
            "eigenvalues": eigvals,
            "eigenvectors": W,
            "explained_variance_ratio": explained,
            "mean": mean.ravel(),
        },
    )


def cheatsheet() -> str:
    return "pcasg({}) -> PCA signal decomposition."
