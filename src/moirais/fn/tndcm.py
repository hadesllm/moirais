"""CP tensor decomposition via ALS."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def tensor_decompose(X, rank: int = 3, max_iter: int = 100, tol: float = 1e-6, **kwargs) -> DescriptiveResult:
    """CP tensor decomposition via alternating least squares.

    Parameters
    ----------
    X : array-like, shape (I, J, K)
        3-D input tensor.
    rank : int
        Number of components (default 3).
    max_iter : int
        Maximum ALS iterations (default 100).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is final reconstruction error; ``extra`` has ``factors``
        (list of 3 factor matrices), ``iterations``, ``weights``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("Input must be a 3-D tensor")
    I, J, K = X.shape
    rng = np.random.default_rng(0)
    A = rng.standard_normal((I, rank))
    B = rng.standard_normal((J, rank))
    C = rng.standard_normal((K, rank))

    X0 = X.reshape(I, J * K)
    X1 = X.transpose(1, 0, 2).reshape(J, I * K)
    X2 = X.transpose(2, 0, 1).reshape(K, I * J)

    iters = 0
    for it in range(max_iter):
        CB = np.array([np.kron(C[:, r], B[:, r]) for r in range(rank)]).T
        A_new = X0 @ CB @ np.linalg.pinv(CB.T @ CB)
        A = A_new

        CA = np.array([np.kron(C[:, r], A[:, r]) for r in range(rank)]).T
        B_new = X1 @ CA @ np.linalg.pinv(CA.T @ CA)
        B = B_new

        BA = np.array([np.kron(B[:, r], A[:, r]) for r in range(rank)]).T
        C_new = X2 @ BA @ np.linalg.pinv(BA.T @ BA)
        C = C_new

        recon = np.zeros_like(X)
        for r in range(rank):
            recon += np.einsum("i,j,k->ijk", A[:, r], B[:, r], C[:, r])
        err = np.linalg.norm(X - recon) / (np.linalg.norm(X) + 1e-15)
        iters = it + 1
        if err < tol:
            break

    weights = np.array(
        [np.linalg.norm(A[:, r]) * np.linalg.norm(B[:, r]) * np.linalg.norm(C[:, r]) for r in range(rank)]
    )
    return DescriptiveResult(
        name="tensor_decompose",
        value=float(err),
        extra={"factors": [A, B, C], "iterations": iters, "weights": weights},
    )


tndcm = tensor_decompose


def cheatsheet() -> str:
    return "tensor_decompose({}) -> CP tensor decomposition via ALS."
