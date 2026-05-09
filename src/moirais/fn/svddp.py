"""Singular Value Decomposition."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Nature does not hurry, yet everything is accomplished. — Lao Tzu"


def svd_decompose(A, **kwargs) -> DescriptiveResult:
    """
    Compute the Singular Value Decomposition A = U S V^T.

    :param A: (m, n) matrix.
    :return: DescriptiveResult with singular values and matrix dimensions.

    References
    ----------
    Golub GH, Van Loan CF (2013). Matrix Computations, 4th ed.
    Johns Hopkins University Press.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(S[0] * max(A.shape) * np.finfo(float).eps < S))
    total = float(np.sum(S**2))
    explained = (S**2) / total if total > 0 else S**2
    return DescriptiveResult(
        name="svd_decompose",
        value=float(S[0]),
        extra={
            "singular_values": S.tolist(),
            "explained_variance_ratio": explained.tolist(),
            "rank": rank,
            "shape": list(A.shape),
            "condition_number": float(S[0] / S[-1]) if S[-1] > 0 else float("inf"),
        },
    )


svddp = svd_decompose


def cheatsheet() -> str:
    return "svd_decompose({}) -> Singular Value Decomposition."
