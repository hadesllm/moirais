"""Tucker tensor decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


def tucker_decompose(X, ranks=(2, 2, 2), max_iter: int = 100, tol: float = 1e-6, **kwargs) -> DescriptiveResult:
    """Tucker tensor decomposition via HOSVD + ALS refinement.

    Parameters
    ----------
    X : array-like, shape (I, J, K)
        3-D input tensor.
    ranks : tuple of int
        Multilinear ranks (R1, R2, R3) (default (2,2,2)).
    max_iter : int
        Maximum ALS iterations (default 100).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is reconstruction error; ``extra`` has ``core``
        (R1 x R2 x R3), ``factors`` (list of 3 factor matrices),
        ``iterations``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("Input must be a 3-D tensor")
    I, J, K = X.shape
    R1, R2, R3 = ranks

    def _unfold(T, mode):
        return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)

    X0 = _unfold(X, 0)
    U0, _, _ = np.linalg.svd(X0, full_matrices=False)
    A = U0[:, :R1]

    X1 = _unfold(X, 1)
    U1, _, _ = np.linalg.svd(X1, full_matrices=False)
    B = U1[:, :R2]

    X2 = _unfold(X, 2)
    U2, _, _ = np.linalg.svd(X2, full_matrices=False)
    C = U2[:, :R3]

    iters = 0
    for it in range(max_iter):
        Y1 = np.einsum("ijk,jr,ks->irs", X, B, C)
        U, _, _ = np.linalg.svd(_unfold(Y1, 0), full_matrices=False)
        A_new = U[:, :R1]

        Y2 = np.einsum("ijk,ir,ks->rjs", X, A_new, C)
        U, _, _ = np.linalg.svd(Y2.reshape(J, -1), full_matrices=False)
        B_new = U[:, :R2]

        Y3 = np.einsum("ijk,ir,js->rsk", X, A_new, B_new)
        U, _, _ = np.linalg.svd(Y3.reshape(K, -1), full_matrices=False)
        C_new = U[:, :R3]

        delta = np.linalg.norm(A_new - A) + np.linalg.norm(B_new - B) + np.linalg.norm(C_new - C)
        A, B, C = A_new, B_new, C_new
        iters = it + 1
        if delta < tol:
            break

    core = np.einsum("ijk,ia,jb,kc->abc", X, A, B, C)
    recon = np.einsum("abc,ia,jb,kc->ijk", core, A, B, C)
    err = float(np.linalg.norm(X - recon) / (np.linalg.norm(X) + 1e-15))

    return DescriptiveResult(
        name="tucker_decompose",
        value=err,
        extra={"core": core, "factors": [A, B, C], "iterations": iters},
    )


tkcmp = tucker_decompose


def cheatsheet() -> str:
    return "tucker_decompose({}) -> Tucker tensor decomposition."
