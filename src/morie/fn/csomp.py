# morie.fn -- function file (hadesllm/morie)
"""Orthogonal Matching Pursuit for sparse recovery."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I cannot teach anybody anything. I can only make them think. -- Socrates"


def omp(A, y, sparsity: int, **kwargs) -> DescriptiveResult:
    r"""
    Orthogonal Matching Pursuit (OMP) for sparse signal recovery.

    Greedily selects columns of :math:`A` that best explain :math:`y`,
    solving :math:`\\min \\|x\\|_0` subject to :math:`Ax = y`.

    At each step, selects the atom most correlated with the residual,
    then solves the least-squares subproblem on the selected support.

    :param A: (m, n) sensing/dictionary matrix.
    :param y: (m,) observation vector.
    :param sparsity: Target sparsity level (number of nonzeros).
    :return: DescriptiveResult with recovered sparse vector.
    :raises ValueError: If sparsity > m or dimensions mismatch.

    References
    ----------
    Pati, Y. C., Rezaiifar, R. & Krishnaprasad, P. S. (1993). Orthogonal
    matching pursuit: recursive function approximation with applications
    to wavelet decomposition. *Asilomar Conference*, 1, 40-44.
    Tropp, J. A. & Gilbert, A. C. (2007). Signal recovery from random
    measurements via Orthogonal Matching Pursuit. *IEEE Trans. Inform.
    Theory*, 53(12), 4655-4666.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    m, n = A.shape

    if len(y) != m:
        raise ValueError(f"y length ({len(y)}) must match A rows ({m}).")
    if sparsity > m:
        raise ValueError(f"sparsity ({sparsity}) must be <= m ({m}).")

    residual = y.copy()
    support = []
    x = np.zeros(n)

    for _ in range(sparsity):
        correlations = np.abs(A.T @ residual)
        correlations[support] = -1
        idx = int(np.argmax(correlations))
        support.append(idx)

        A_s = A[:, support]
        x_s, _, _, _ = np.linalg.lstsq(A_s, y, rcond=None)
        residual = y - A_s @ x_s

    x_full = np.zeros(n)
    for i, s in enumerate(support):
        x_full[s] = x_s[i]

    residual_norm = float(np.linalg.norm(residual))

    return DescriptiveResult(
        name="omp",
        value=residual_norm,
        extra={
            "x_recovered": x_full,
            "support": np.array(support),
            "residual_norm": residual_norm,
            "sparsity": sparsity,
            "nnz": int(np.sum(x_full != 0)),
        },
    )


csomp = omp


def cheatsheet() -> str:
    return "omp({}) -> Orthogonal Matching Pursuit for sparse recovery."
