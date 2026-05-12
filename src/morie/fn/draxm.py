# morie.fn — function file (hadesllm/morie)
"""Matrix decomposition (destructive rank reduction). 'Nothing goes over my head.' -- Drax"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def destroyer_decompose(
    A: np.ndarray,
    *,
    rank: int | None = None,
    tol: float = 1e-10,
) -> DescriptiveResult:
    r"""Truncated SVD with rank estimation and reconstruction error.

    Decomposes :math:`A \\approx U_k \\Sigma_k V_k^T` and reports the
    Frobenius-norm reconstruction error and effective rank.

    Parameters
    ----------
    A : np.ndarray
        Input matrix (m x n).
    rank : int or None
        Target rank. If None, estimated via singular value decay.
    tol : float
        Threshold for effective rank determination.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``U``, ``S``, ``Vt``, ``rank``,
        ``reconstruction_error``, ``energy_retained``.
    """
    M = np.asarray(A, dtype=float)
    if M.ndim != 2:
        raise ValueError("A must be 2D")

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    if rank is None:
        rank = int(np.sum(tol * S[0] < S))
    rank = max(1, min(rank, len(S)))

    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]

    reconstructed = U_k * S_k[None, :] @ Vt_k
    error = float(np.linalg.norm(M - reconstructed, "fro"))
    total_energy = float(np.sum(S**2))
    retained = float(np.sum(S_k**2) / total_energy) if total_energy > 0 else 1.0

    return DescriptiveResult(
        name="destroyer_decompose",
        value={
            "U": U_k,
            "S": S_k,
            "Vt": Vt_k,
            "rank": rank,
            "reconstruction_error": error,
            "energy_retained": retained,
        },
        extra={"m": M.shape[0], "n": M.shape[1], "full_rank": len(S)},
    )


draxm = destroyer_decompose


def cheatsheet() -> str:
    return "destroyer_decompose({}) -> Matrix decomposition (destructive rank reduction). 'Nothing "
