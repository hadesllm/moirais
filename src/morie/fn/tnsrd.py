"""CP Tensor Decomposition via Alternating Least Squares."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Difficulties strengthen the mind, as labor does the body. -- Seneca"


def tensor_decompose(
    X, rank: int = 3, max_iter: int = 100, tol: float = 1e-6, seed: int | None = None, **kwargs
) -> DescriptiveResult:
    """CP tensor decomposition via alternating least squares.

    Decomposes a 3-D tensor into a sum of rank-one components.

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
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is relative reconstruction error; ``extra`` has
        ``factors`` (list of 3 factor matrices), ``lambdas`` (weights).

    References
    ----------
    Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and
    applications. *SIAM Rev.*, 51(3), 455-500.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("Input must be a 3-D array.")
    I, J, K = X.shape
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((I, rank))
    B = rng.standard_normal((J, rank))
    C = rng.standard_normal((K, rank))

    def _khatri_rao(A, B):
        I_, R = A.shape
        J_, R2 = B.shape
        return (
            np.array([np.kron(A[i], B[i]) for i in range(R)]).T.reshape(I_ * J_, R)
            if False
            else np.einsum("ir,jr->ijr", A, B).reshape(-1, R)
        )

    X0 = X.reshape(I, J * K)
    X1 = X.transpose(1, 0, 2).reshape(J, I * K)
    X2 = X.transpose(2, 0, 1).reshape(K, I * J)

    for _ in range(max_iter):
        V = (B.T @ B) * (C.T @ C)
        A_new = X0 @ _khatri_rao(C, B) @ np.linalg.pinv(V)
        V = (A_new.T @ A_new) * (C.T @ C)
        B_new = X1 @ _khatri_rao(C, A_new) @ np.linalg.pinv(V)
        V = (A_new.T @ A_new) * (B_new.T @ B_new)
        C_new = X2 @ _khatri_rao(B_new, A_new) @ np.linalg.pinv(V)
        change = np.linalg.norm(A_new - A) + np.linalg.norm(B_new - B) + np.linalg.norm(C_new - C)
        A, B, C = A_new, B_new, C_new
        if change < tol:
            break

    lambdas = np.linalg.norm(A, axis=0)
    A = A / (lambdas[np.newaxis, :] + 1e-15)
    recon = np.zeros_like(X)
    for r in range(rank):
        recon += lambdas[r] * np.einsum("i,j,k", A[:, r], B[:, r], C[:, r])
    rel_err = float(np.linalg.norm(X - recon) / (np.linalg.norm(X) + 1e-15))
    return DescriptiveResult(
        name="tensor_decompose",
        value=rel_err,
        extra={"factors": [A, B, C], "lambdas": lambdas, "rank": rank},
    )


tnsrd = tensor_decompose


def cheatsheet() -> str:
    return "tensor_decompose({}) -> CP Tensor Decomposition via Alternating Least Squares."
