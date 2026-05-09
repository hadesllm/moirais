# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Multi-view learning (CCA fusion). 'The Final Flight of the Osiris.' -- Animatrix"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def multiview_cca(
    X1: np.ndarray,
    X2: np.ndarray,
    *,
    n_components: int = 2,
    reg: float = 1e-4,
) -> DescriptiveResult:
    """Canonical Correlation Analysis between two views.

    Finds linear projections of X1 and X2 that maximise mutual correlation.
    Returns canonical correlations and projected data.

    Parameters
    ----------
    X1 : ndarray
        First view (n x p1).
    X2 : ndarray
        Second view (n x p2). Must have same n rows as X1.
    n_components : int
        Number of canonical components.
    reg : float
        Regularisation added to diagonal of covariance matrices.

    Returns
    -------
    DescriptiveResult
        ``value`` is the first canonical correlation; ``extra`` has all
        canonical correlations and projection matrices.

    References
    ----------
    Hotelling, H. (1936). Relations between two sets of variates.
    Biometrika, 28(3/4), 321-377.
    """
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1 and X2 must be 2-D")
    if X1.shape[0] != X2.shape[0]:
        raise ValueError("X1 and X2 must have the same number of rows")
    n = X1.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 observations")

    X1c = X1 - X1.mean(axis=0)
    X2c = X2 - X2.mean(axis=0)

    C11 = X1c.T @ X1c / (n - 1) + reg * np.eye(X1.shape[1])
    C22 = X2c.T @ X2c / (n - 1) + reg * np.eye(X2.shape[1])
    C12 = X1c.T @ X2c / (n - 1)

    C11_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C11)).T
    C22_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C22)).T

    M = C11_inv_sqrt @ C12 @ C22_inv_sqrt
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    k = min(n_components, len(s))
    canoncorr = s[:k]

    A = C11_inv_sqrt @ U[:, :k]
    B = C22_inv_sqrt @ Vt[:k, :].T

    Z1 = X1c @ A
    Z2 = X2c @ B

    return DescriptiveResult(
        name="Multi-View CCA",
        value=float(canoncorr[0]),
        extra={
            "canonical_correlations": canoncorr.tolist(),
            "n_components": k,
            "n": n,
            "p1": X1.shape[1],
            "p2": X2.shape[1],
            "total_correlation": float(canoncorr.sum()),
        },
    )


anmls = multiview_cca


def cheatsheet() -> str:
    return "multiview_cca({}) -> Multi-view learning (CCA fusion). 'The Final Flight of the O"
