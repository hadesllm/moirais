# morie.fn — function file (hadesllm/morie)
"""Channel capacity for discrete memoryless channels."""

__all__ = ["chcap"]

import numpy as np
from ._richresult import RichResult


def chcap(
    transition_matrix: np.ndarray,
    *,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> dict:
    """
    Compute channel capacity C = max_{p(x)} I(X;Y) via Blahut-Arimoto.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Channel matrix p(y|x), shape (n_input, n_output). Rows sum to 1.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        'capacity' (bits), 'optimal_input' (optimal input distribution).

    Raises
    ------
    ValueError
        If transition_matrix is invalid.

    References
    ----------
    Arimoto, S. (1972). IEEE Trans. Inform. Theory, 18(1), 14-20.
    Cover & Thomas (2006). Elements of Information Theory, Ch. 7.
    """
    W = np.asarray(transition_matrix, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("transition_matrix must be 2-D.")
    if not np.allclose(W.sum(axis=1), 1.0):
        raise ValueError("Rows of transition_matrix must sum to 1.")
    if np.any(W < 0):
        raise ValueError("transition_matrix entries must be non-negative.")

    n, m = W.shape
    r = np.ones(n) / n
    eps = 1e-300

    for _ in range(max_iter):
        q = r @ W
        log_ratio = np.zeros((n, m))
        for j in range(m):
            if q[j] > eps:
                log_ratio[:, j] = np.log2(W[:, j] / (q[j] + eps) + eps)
        c_vec = np.exp2(np.sum(W * log_ratio, axis=1))
        r_new = r * c_vec
        r_new /= r_new.sum()
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new

    q = r @ W
    mi = 0.0
    for i in range(n):
        for j in range(m):
            if r[i] > eps and W[i, j] > eps and q[j] > eps:
                mi += r[i] * W[i, j] * np.log2(W[i, j] / q[j])

    return RichResult(payload={"capacity": max(mi, 0.0), "optimal_input": r})
