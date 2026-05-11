# morie.fn — function file (hadesllm/morie)
"""Orthogonal Matching Pursuit sparse approximation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In my experience there is no such thing as luck."


def omp_sparse(D, x, sparsity: int = 10, **kwargs) -> DescriptiveResult:
    """Orthogonal Matching Pursuit for sparse signal approximation.

    Greedily selects *sparsity* atoms from dictionary *D* to approximate *x*.

    Parameters
    ----------
    D : array-like, shape (n, m)
        Dictionary matrix (columns are atoms).
    x : array-like, shape (n,)
        Signal to approximate.
    sparsity : int
        Maximum number of non-zero coefficients (default 10).

    Returns
    -------
    DescriptiveResult
        ``value`` is reconstruction error; ``extra`` has ``coeffs``,
        ``support``, ``reconstruction``.

    References
    ----------
    Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993).
    Orthogonal matching pursuit: recursive function approximation with
    applications to wavelet decomposition. *Asilomar Conf.*, 1, 40-44.
    """
    D = np.asarray(D, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    n, m = D.shape
    residual = x.copy()
    support = []
    coeffs = np.zeros(m)
    for _ in range(min(sparsity, m)):
        correlations = D.T @ residual
        idx = int(np.argmax(np.abs(correlations)))
        if idx in support:
            break
        support.append(idx)
        Ds = D[:, support]
        c, _, _, _ = np.linalg.lstsq(Ds, x, rcond=None)
        residual = x - Ds @ c
    for i, s in enumerate(support):
        coeffs[s] = c[i]
    reconstruction = D @ coeffs
    err = float(np.linalg.norm(x - reconstruction))
    return DescriptiveResult(
        name="omp_sparse",
        value=err,
        extra={"coeffs": coeffs, "support": support, "reconstruction": reconstruction, "sparsity": len(support)},
    )


ompfn = omp_sparse


def cheatsheet() -> str:
    return "omp_sparse({}) -> Orthogonal Matching Pursuit sparse approximation."
