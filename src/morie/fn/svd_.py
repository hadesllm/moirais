"""Truncated SVD."""

import numpy as np

from ._containers import DescriptiveResult


def truncated_svd(X: np.ndarray, k: int = 2) -> DescriptiveResult:
    """
    Truncated Singular Value Decomposition.

    :param X: (n, p) data matrix.
    :param k: Number of singular values/vectors to retain.
    :return: DescriptiveResult with U, S, Vt components.

    References
    ----------
    Golub GH, Van Loan CF (2013). Matrix Computations. 4th ed.
    Johns Hopkins University Press.
    """
    X = np.asarray(X, dtype=np.float64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, len(s))
    total_energy = np.sum(s**2)
    retained_energy = np.sum(s[:k] ** 2)
    reconstruction = U[:, :k] * s[:k] @ Vt[:k]
    frob_error = float(np.linalg.norm(X - reconstruction, "fro"))
    return DescriptiveResult(
        name="truncated_svd",
        value=float(retained_energy / total_energy),
        extra={
            "U": U[:, :k],
            "S": s[:k],
            "Vt": Vt[:k],
            "energy_ratio": float(retained_energy / total_energy),
            "frobenius_error": frob_error,
            "k": k,
        },
    )


svd_ = truncated_svd


def cheatsheet() -> str:
    return "truncated_svd({}) -> Truncated SVD."
