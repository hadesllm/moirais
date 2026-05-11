"""Data whitening (sphering)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def whitening(X, **kwargs) -> DescriptiveResult:
    """Whiten (sphere) data to zero mean and identity covariance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix.

    Returns
    -------
    DescriptiveResult
        ``value`` is condition number of whitening transform; ``extra``
        has ``X_white``, ``whitening_matrix``, ``mean``.

    References
    ----------
    Kessy, A., Lewin, A., & Strimmer, K. (2018). Optimal whitening and
    decorrelation. *Am. Stat.*, 72(4), 309-314.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    D_inv = np.diag(1.0 / np.sqrt(eigvals + 1e-10))
    W = D_inv @ eigvecs.T
    X_white = (W @ Xc.T).T
    cond = float(np.max(eigvals) / (np.min(eigvals) + 1e-15))
    return DescriptiveResult(
        name="whitening",
        value=cond,
        extra={"X_white": X_white, "whitening_matrix": W, "mean": mean},
    )


whtfn = whitening


def cheatsheet() -> str:
    return "whitening({}) -> Data whitening (sphering)."
