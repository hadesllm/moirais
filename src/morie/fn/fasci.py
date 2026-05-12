# morie.fn -- function file (hadesllm/morie)
"""FastICA algorithm for blind source separation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Train yourself to let go of everything you fear to lose."


def fastica(
    X, n_components: int | None = None, max_iter: int = 200, tol: float = 1e-4, seed: int | None = None, **kwargs
) -> DescriptiveResult:
    """FastICA: independent component analysis via fixed-point iteration.

    Uses the deflation approach with log-cosh non-linearity.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix (observations as rows).
    n_components : int or None
        Number of independent components. Default: n_features.
    max_iter : int
        Maximum iterations per component (default 200).
    tol : float
        Convergence tolerance (default 1e-4).
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is n_components; ``extra`` has ``sources``,
        ``mixing_matrix``, ``unmixing_matrix``.

    References
    ----------
    Hyvarinen, A. (1999). Fast and robust fixed-point algorithms for
    independent component analysis. *IEEE Trans. Neural Netw.*, 10(3),
    626-634.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n_components is None:
        n_components = p
    n_components = min(n_components, p)
    rng = np.random.default_rng(seed)

    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    D = np.diag(1.0 / np.sqrt(eigvals[:n_components] + 1e-10))
    K = D @ eigvecs[:, :n_components].T
    Xw = (K @ Xc.T).T

    W = np.zeros((n_components, n_components))
    for i in range(n_components):
        w = rng.standard_normal(n_components)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            wx = Xw @ w
            g = np.tanh(wx)
            gp = 1.0 - g**2
            w_new = (Xw.T @ g) / n - np.mean(gp) * w
            for j in range(i):
                w_new -= np.dot(w_new, W[j]) * W[j]
            w_new /= np.linalg.norm(w_new) + 1e-15
            if abs(abs(np.dot(w_new, w)) - 1.0) < tol:
                w = w_new
                break
            w = w_new
        W[i] = w

    sources = Xw @ W.T
    unmixing = W @ K
    mixing = np.linalg.pinv(unmixing)
    return DescriptiveResult(
        name="fastica",
        value=n_components,
        extra={"sources": sources, "mixing_matrix": mixing, "unmixing_matrix": unmixing},
    )


fasci = fastica


def cheatsheet() -> str:
    return "fastica({}) -> FastICA algorithm for blind source separation."
