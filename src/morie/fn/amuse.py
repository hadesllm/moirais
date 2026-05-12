# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""AMUSE algorithm for blind source separation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Around the survivors a perimeter create."


def amuse_bss(X, lag: int = 1, **kwargs) -> DescriptiveResult:
    """AMUSE: Algorithm for Multiple Unknown Signals Extraction.

    Simplified SOBI using a single time lag. Whitens the data, computes
    a time-lagged covariance, and diagonalises it via SVD.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_channels)
        Mixed signal matrix.
    lag : int
        Time lag for covariance estimation (default 1).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of sources; ``extra`` has ``sources``,
        ``mixing_matrix``, ``unmixing_matrix``.

    References
    ----------
    Tong, L., Liu, R., Soon, V. C., & Huang, Y.-F. (1991). Indeterminacy
    and identifiability of blind identification. *IEEE Trans. CAS*,
    38(5), 499-509.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean

    cov0 = Xc.T @ Xc / n
    eigvals, eigvecs = np.linalg.eigh(cov0)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    D = np.diag(1.0 / np.sqrt(eigvals + 1e-10))
    W = D @ eigvecs.T
    Z = (W @ Xc.T).T

    Rlag = Z[lag:].T @ Z[:-lag] / (n - lag)
    Rlag = (Rlag + Rlag.T) / 2.0
    U, _, Vt = np.linalg.svd(Rlag)

    unmixing = U.T @ W
    sources = Xc @ unmixing.T
    mixing = np.linalg.pinv(unmixing)
    return DescriptiveResult(
        name="amuse_bss",
        value=p,
        extra={"sources": sources, "mixing_matrix": mixing, "unmixing_matrix": unmixing, "lag": lag},
    )


amuse = amuse_bss


def cheatsheet() -> str:
    return "amuse_bss({}) -> AMUSE algorithm for blind source separation."
