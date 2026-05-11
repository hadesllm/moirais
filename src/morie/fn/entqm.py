# morie.fn — function file (hadesllm/morie)
"""Quantum entanglement entropy (von Neumann)."""

__all__ = ["entqm"]

import numpy as np


def entqm(
    rho: np.ndarray,
    base: float = np.e,
) -> dict:
    """
    Compute the von Neumann entropy of a quantum density matrix.

    .. math::

        S(\\rho) = -\\text{Tr}(\\rho \\ln \\rho)
                 = -\\sum_i \\lambda_i \\ln \\lambda_i

    where :math:`\\lambda_i` are eigenvalues of :math:`\\rho`.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (Hermitian, positive semi-definite, trace 1).
    base : float
        Logarithm base. Default e (nats). Use 2 for bits.

    Returns
    -------
    dict
        Keys: entropy, eigenvalues, purity (Tr(rho^2)),
        is_pure (bool), rank.
    """
    rho = np.asarray(rho, dtype=complex)
    n = rho.shape[0]
    if rho.shape != (n, n):
        raise ValueError("rho must be square.")

    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.real(eigvals)
    eigvals = np.clip(eigvals, 0, None)

    mask = eigvals > 1e-15
    S = 0.0
    for lam in eigvals[mask]:
        S -= lam * np.log(lam) / np.log(base)

    purity = float(np.real(np.trace(rho @ rho)))
    rank = int(np.sum(mask))

    return {
        "entropy": float(S),
        "eigenvalues": eigvals,
        "purity": purity,
        "is_pure": abs(purity - 1.0) < 1e-10,
        "rank": rank,
    }
