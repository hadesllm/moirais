# moirais.fn — function file (hadesllm/moirais)
"""Exploratory Factor Analysis via principal axis factoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import FaRes


def _varimax(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Varimax rotation (Kaiser, 1958)."""
    p, k = loadings.shape
    R = np.eye(k)
    L = loadings.copy()
    for _ in range(max_iter):
        LR = L @ R
        # Varimax criterion
        D = (LR**3) - LR * (np.sum(LR**2, axis=0) / p)
        U, s, Vt = np.linalg.svd(L.T @ D)
        R_new = U @ Vt
        if np.max(np.abs(R_new - R)) < tol:
            break
        R = R_new
    return L @ R


def _parallel_analysis(data: np.ndarray, n_sim: int = 100, rng: np.random.Generator | None = None) -> int:
    """Determine number of factors via Horn's parallel analysis."""
    if rng is None:
        rng = np.random.default_rng(0)
    n, p = data.shape
    R = np.corrcoef(data, rowvar=False)
    real_eigvals = np.sort(np.linalg.eigvalsh(R))[::-1]

    sim_eigvals = np.zeros((n_sim, p))
    for i in range(n_sim):
        sim = rng.standard_normal((n, p))
        Rsim = np.corrcoef(sim, rowvar=False)
        sim_eigvals[i] = np.sort(np.linalg.eigvalsh(Rsim))[::-1]

    threshold = np.percentile(sim_eigvals, 95, axis=0)
    n_factors = int(np.sum(real_eigvals > threshold))
    return max(n_factors, 1)


def fa(
    data: pd.DataFrame | np.ndarray,
    n_factors: int | None = None,
    rotation: str = "varimax",
    max_iter: int = 100,
    tol: float = 1e-4,
) -> FaRes:
    """Exploratory Factor Analysis via principal axis factoring.

    Iterates: communalities -> reduced correlation -> eigendecompose ->
    update communalities, until convergence.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    n_factors : int, optional
        Number of factors.  *None* uses parallel analysis.
    rotation : str
        ``"varimax"`` (default) or ``"none"``.
    max_iter : int
        Maximum PAF iterations.
    tol : float
        Convergence tolerance on communalities.

    Returns
    -------
    FaRes
        ``loadings`` (p x k), ``communalities``, ``eigenvalues``,
        ``variance_explained`` (per-factor proportion).

    References
    ----------
    Fabrigar, L. R. et al. (1999). Evaluating the use of exploratory factor
    analysis in psychological research. *Psychological Methods*, 4(3), 272-299.
    DOI: 10.1037/1082-989X.4.3.272
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape

    R = np.corrcoef(X, rowvar=False)

    if n_factors is None:
        n_factors = _parallel_analysis(X)
    k = min(n_factors, p)

    # Initial communalities: squared multiple correlations
    try:
        Ri = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Ri = np.linalg.pinv(R)
    h2 = 1.0 - 1.0 / np.diag(Ri)
    h2 = np.clip(h2, 0.01, 0.99)

    for _ in range(max_iter):
        R_reduced = R.copy()
        np.fill_diagonal(R_reduced, h2)

        eigvals, eigvecs = np.linalg.eigh(R_reduced)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Keep only k factors with positive eigenvalues
        eigvals_k = np.maximum(eigvals[:k], 0.0)
        loadings = eigvecs[:, :k] * np.sqrt(eigvals_k)

        h2_new = np.sum(loadings**2, axis=1)
        h2_new = np.clip(h2_new, 0.01, 0.99)

        if np.max(np.abs(h2_new - h2)) < tol:
            h2 = h2_new
            break
        h2 = h2_new

    # Apply rotation
    if rotation == "varimax" and k > 1:
        loadings = _varimax(loadings)

    # Variance explained
    var_explained = np.sum(loadings**2, axis=0) / p

    # Full eigenvalues of original correlation
    all_eigvals = np.sort(np.linalg.eigvalsh(R))[::-1]

    return FaRes(
        loadings=loadings,
        communalities=h2,
        eigenvalues=all_eigvals,
        variance_explained=var_explained,
        n_factors=k,
        rotation=rotation,
    )


def cheatsheet() -> str:
    return "_varimax({}) -> Exploratory Factor Analysis via principal axis factoring."
