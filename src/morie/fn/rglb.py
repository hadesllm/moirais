# morie.fn -- function file (hadesllm/morie)
"""Greatest Lower Bound on reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rglb(
    data: pd.DataFrame | np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> float:
    """Greatest Lower Bound (GLB) on reliability.

    The GLB is the maximum of Lambda 4 over all possible splits, which
    equals the maximum reliability consistent with the observed
    covariance matrix.  Computed via the eigenvalue method of
    Woodhouse and Jackson (1977): iteratively maximise the sum of
    unique variances subject to the constraint that the residual
    covariance matrix (C - diag(u)) remains positive semi-definite.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).
    max_iter : int
        Maximum iterations for the optimisation (default 1000).
    tol : float
        Convergence tolerance (default 1e-8).

    Returns
    -------
    float
        GLB reliability estimate.

    References
    ----------
    Woodhouse, B., & Jackson, P. H. (1977). Lower bounds for the
    reliability of the total score on a test composed of non-homogeneous
    items: II: A search procedure to locate the greatest lower bound.
    *Psychometrika*, 42(4), 579-591.

    Jackson, P. H., & Agunwamba, C. C. (1977). Lower bounds for the
    reliability of the total score on a test composed of non-homogeneous
    items: I: Algebraic lower bounds. *Psychometrika*, 42(4), 567-578.
    """
    X = np.asarray(data, dtype=np.float64)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, k = X.shape

    if k < 2:
        return float("nan")

    C = np.cov(X, rowvar=False, ddof=1)
    total_var = C.sum()

    if total_var < 1e-15:
        return float("nan")

    # Initialise unique variances as item variances
    u = np.diag(C).copy()

    for _ in range(max_iter):
        u_old = u.copy()

        # Reduced covariance matrix
        C_reduced = C - np.diag(u)

        # Eigendecomposition
        eigvals = np.linalg.eigvalsh(C_reduced)

        # Set negative eigenvalues to zero (PSD projection)
        eigvals_clipped = np.maximum(eigvals, 0.0)

        # Reconstruct communality matrix
        eigvecs_full = np.linalg.eigh(C_reduced)[1]
        C_psd = eigvecs_full @ np.diag(eigvals_clipped) @ eigvecs_full.T

        # Update unique variances: u_j = c_jj - psd_jj
        u_new = np.diag(C) - np.diag(C_psd)
        # Ensure non-negative unique variances bounded by item variance
        u = np.clip(u_new, 0.0, np.diag(C))

        if np.max(np.abs(u - u_old)) < tol:
            break

    glb = 1.0 - u.sum() / total_var
    return float(min(glb, 1.0))


short = rglb


def cheatsheet() -> str:
    return "rglb({}) -> Greatest Lower Bound on reliability."
