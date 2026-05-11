# morie.fn — function file (hadesllm/morie)
"""Projection Pursuit (maximise non-Gaussianity)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Difficult to see. Always in motion is the future."


def projection_pursuit(
    X, n_components: int = 2, max_iter: int = 200, tol: float = 1e-5, seed: int | None = None, **kwargs
) -> DescriptiveResult:
    """Projection pursuit: find projections maximising non-Gaussianity.

    Uses negentropy approximation via log-cosh to find interesting
    projection directions in a deflationary scheme.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    n_components : int
        Number of projection directions (default 2).
    max_iter : int
        Maximum iterations per direction (default 200).
    tol : float
        Convergence tolerance (default 1e-5).
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is n_components; ``extra`` has ``projections``,
        ``directions``, ``negentropies``.

    References
    ----------
    Friedman, J. H. (1987). Exploratory projection pursuit. *J. Am.
    Stat. Assoc.*, 82(397), 249-266.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    n_components = min(n_components, p)
    rng = np.random.default_rng(seed)

    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    D_inv = np.diag(1.0 / np.sqrt(eigvals + 1e-10))
    Xw = Xc @ eigvecs @ D_inv

    directions = []
    negentropies = []
    for _ in range(n_components):
        w = rng.standard_normal(p)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            proj = Xw @ w
            g = np.tanh(proj)
            gp = 1.0 - g**2
            w_new = (Xw.T @ g) / n - np.mean(gp) * w
            for d in directions:
                w_new -= np.dot(w_new, d) * d
            w_new /= np.linalg.norm(w_new) + 1e-15
            if abs(abs(np.dot(w_new, w)) - 1.0) < tol:
                w = w_new
                break
            w = w_new
        directions.append(w)
        proj = Xw @ w
        neg_g = np.mean(np.log(np.cosh(proj)))
        neg_ref = np.mean(np.log(np.cosh(rng.standard_normal(n))))
        negentropies.append(float((neg_g - neg_ref) ** 2))

    directions = np.array(directions)
    projections = Xw @ directions.T
    return DescriptiveResult(
        name="projection_pursuit",
        value=n_components,
        extra={"projections": projections, "directions": directions, "negentropies": negentropies},
    )


projp = projection_pursuit


def cheatsheet() -> str:
    return "projection_pursuit({}) -> Projection Pursuit (maximise non-Gaussianity)."
