# morie.fn — function file (hadesllm/morie)
"""ICA-based source separation (FastICA).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 16.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['icasg']

_QUOTE = "Separate the light from the dark. -- Mace Windu"


def icasg(
    X: np.ndarray,
    *,
    n_components: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 42,
) -> DescriptiveResult:
    """FastICA source separation.

    Parameters
    ----------
    X : array-like
        2-D array (n_channels, n_samples).
        If more columns than rows, assumed (n_channels, n_samples).
    n_components : int or None
        Number of independent components (default n_channels).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
    """
    rng = np.random.default_rng(seed)
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
    eigvals = eigvals[idx][:n_components]
    eigvecs = eigvecs[:, idx][:, :n_components]
    D_inv = np.diag(1.0 / np.sqrt(eigvals + 1e-12))
    K = D_inv @ eigvecs.T
    Xw = K @ Xc

    W = rng.standard_normal((n_components, n_components))
    W, _ = np.linalg.qr(W)

    for _ in range(max_iter):
        WX = W @ Xw
        g = np.tanh(WX)
        gp = 1 - g ** 2
        W_new = g @ Xw.T / n_samp - gp.mean(axis=1, keepdims=True) * W
        W_new, _ = np.linalg.qr(W_new)
        if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1)) < tol:
            W = W_new
            break
        W = W_new

    sources = W @ Xw
    mixing = np.linalg.pinv(W @ K)

    return DescriptiveResult(
        name="icasg",
        value=float(n_components),
        extra={
            "sources": sources,
            "mixing_matrix": mixing,
            "unmixing_matrix": W @ K,
            "mean": mean.ravel(),
        },
    )


def cheatsheet() -> str:
    return "icasg({}) -> ICA source separation (FastICA)."
