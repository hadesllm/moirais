# morie.fn — function file (hadesllm/morie)
"""Classical (metric) multidimensional scaling."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def metric_mds(D: np.ndarray, n_dims: int = 2) -> DescriptiveResult:
    """Classical MDS (Torgerson scaling) from a distance matrix.

    Parameters
    ----------
    D : (n, n) symmetric distance matrix
    n_dims : int
        Embedding dimensions.

    Returns
    -------
    DescriptiveResult
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("D must be square.")

    D2 = D**2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:n_dims]
    eigvecs = eigvecs[:, idx][:, :n_dims]

    eigvals = np.maximum(eigvals, 0)
    coords = eigvecs * np.sqrt(eigvals)[None, :]

    stress = 0.0
    D_hat = np.sqrt(np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2))
    mask = np.triu_indices(n, k=1)
    denom = np.sum(D[mask] ** 2)
    if denom > 0:
        stress = np.sqrt(np.sum((D[mask] - D_hat[mask]) ** 2) / denom)

    return DescriptiveResult(
        name="mds",
        value=float(stress),
        extra={"eigenvalues": eigvals.tolist(), "n": n, "n_dims": n_dims, "coords_shape": list(coords.shape)},
    )


mds_ = metric_mds


def cheatsheet() -> str:
    return "metric_mds({}) -> Classical (metric) multidimensional scaling."
