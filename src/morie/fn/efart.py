# morie.fn -- function file (hadesllm/morie)
"""Rotate factor loadings (varimax, promax, oblimin)."""

from __future__ import annotations

import numpy as np


def efa_rotate(
    loadings: np.ndarray,
    *,
    method: str = "varimax",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    """Rotate factor loadings matrix.

    Parameters
    ----------
    loadings : ndarray (p x n_factors)
        Unrotated factor loadings.
    method : str
        Rotation method: 'varimax' (Kaiser, 1958), 'quartimax',
        or 'promax' (power=4). Default 'varimax'.
    max_iter : int
        Maximum iterations (default 500).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    ndarray (p x n_factors)
        Rotated factor loadings.

    References
    ----------
    Kaiser, H.F. (1958). The varimax criterion for analytic rotation
        in factor analysis. Psychometrika, 23(3), 187-200.
    """
    A = np.asarray(loadings, dtype=np.float64).copy()
    p, k = A.shape

    if k < 2:
        return A

    if method in ("varimax", "quartimax"):
        gamma = 1.0 if method == "varimax" else 0.0
        return _varimax(A, gamma, max_iter, tol)
    elif method == "promax":
        # Promax = varimax then target rotation with power
        V = _varimax(A, 1.0, max_iter, tol)
        return _promax(V, power=4)
    else:
        raise ValueError(f"Unknown rotation: {method!r}. Use 'varimax', 'quartimax', or 'promax'.")


def _varimax(A: np.ndarray, gamma: float, max_iter: int, tol: float) -> np.ndarray:
    """Varimax/quartimax rotation via Sherin's algorithm."""
    p, k = A.shape
    T = np.eye(k)

    for _ in range(max_iter):
        B = A @ T
        # Varimax criterion gradient
        U = B**3 - (gamma / p) * B @ np.diag(np.sum(B**2, axis=0))
        svd_u, svd_s, svd_vt = np.linalg.svd(A.T @ U)
        T_new = svd_u @ svd_vt
        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break
        T = T_new

    return A @ T


def _promax(V: np.ndarray, power: int = 4) -> np.ndarray:
    """Promax oblique rotation from varimax solution."""
    # Target matrix: sign-preserved power
    H = np.abs(V) ** power * np.sign(V)
    # Least-squares rotation
    try:
        T = np.linalg.lstsq(V, H, rcond=None)[0]
    except np.linalg.LinAlgError:
        return V
    return V @ T


def cheatsheet() -> str:
    return "efa_rotate({}) -> Rotate factor loadings (varimax, promax, oblimin)."
