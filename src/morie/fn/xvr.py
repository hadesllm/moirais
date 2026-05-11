"""Brain connectivity via partial correlation matrix. 'To me, my X-Men.' -- Professor X"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def partial_corr_matrix(
    data: np.ndarray,
    *,
    regularize: float = 1e-6,
) -> DescriptiveResult:
    """Compute the partial correlation matrix (brain connectivity proxy).

    The partial correlation between variables *i* and *j* given all others
    is derived from the precision matrix (inverse covariance):
    :math:`\\rho_{ij|\\text{rest}} = -\\frac{\\Theta_{ij}}{\\sqrt{\\Theta_{ii} \\Theta_{jj}}}`

    Parameters
    ----------
    data : np.ndarray
        (n x p) data matrix where columns represent brain regions / nodes.
    regularize : float
        Ridge regularization added to diagonal of covariance.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``partial_corr`` (p x p), ``precision`` (p x p),
        ``mean_connectivity``, ``density`` (fraction of |rho| > 0.1).
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be 2D")
    n, p = X.shape
    if n < p + 1:
        raise ValueError("Need n > p for reliable precision matrix")

    X_centered = X - X.mean(axis=0)
    cov = (X_centered.T @ X_centered) / (n - 1) + regularize * np.eye(p)

    precision = np.linalg.inv(cov)

    diag = np.sqrt(np.diag(precision))
    diag[diag == 0] = 1.0
    pcorr = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcorr, 1.0)

    upper = pcorr[np.triu_indices(p, k=1)]
    mean_conn = float(np.mean(np.abs(upper)))
    density = float(np.mean(np.abs(upper) > 0.1))

    return DescriptiveResult(
        name="partial_corr_matrix",
        value={
            "partial_corr": pcorr,
            "precision": precision,
            "mean_connectivity": mean_conn,
            "density": density,
        },
        extra={"n": n, "p": p},
    )


xvr = partial_corr_matrix


def cheatsheet() -> str:
    return "partial_corr_matrix({}) -> Brain connectivity via partial correlation matrix. 'To me, m"
