# morie.fn — function file (hadesllm/morie)
"""Rate-distortion function for discrete memoryless sources."""

__all__ = ["rtdst"]

import numpy as np
from ._richresult import RichResult


def rtdst(
    pmf: np.ndarray,
    distortion_matrix: np.ndarray,
    target_distortion: float,
    *,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> dict:
    r"""
    Compute the rate-distortion function R(D) via Blahut-Arimoto algorithm.

    For source PMF p(x) and distortion d(x, x_hat):

    .. math::

        R(D) = \\min_{q(\\hat{x}|x): E[d] \\le D} I(X; \\hat{X})

    Parameters
    ----------
    pmf : np.ndarray
        Source PMF, shape (n,). Must sum to 1.
    distortion_matrix : np.ndarray
        Distortion d(x, x_hat), shape (n, m).
    target_distortion : float
        Maximum expected distortion D >= 0.
    max_iter : int
        Maximum Blahut-Arimoto iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        'rate' (bits), 'distortion', 'optimal_mapping' (conditional q(x_hat|x)).

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Blahut, R. (1972). IEEE Trans. Inform. Theory, 18(4), 460-473.
    Cover & Thomas (2006). Elements of Information Theory, Ch. 10.
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    distortion_matrix = np.asarray(distortion_matrix, dtype=np.float64)

    if pmf.ndim != 1 or not np.isclose(pmf.sum(), 1.0):
        raise ValueError("pmf must be a 1-D array summing to 1.")
    if distortion_matrix.shape[0] != len(pmf):
        raise ValueError("distortion_matrix rows must match pmf length.")
    if target_distortion < 0:
        raise ValueError("target_distortion must be >= 0.")

    n, m = distortion_matrix.shape
    d_max = np.max(distortion_matrix)

    if target_distortion >= d_max:
        return {"rate": 0.0, "distortion": target_distortion,
                "optimal_mapping": np.ones((n, m)) / m}

    s_val = -5.0
    q_hat = np.ones(m) / m
    eps = 1e-300

    for _ in range(max_iter):
        log_q = np.log(q_hat + eps)
        phi = log_q[None, :] + s_val * distortion_matrix
        phi -= phi.max(axis=1, keepdims=True)
        cond = np.exp(phi)
        cond /= cond.sum(axis=1, keepdims=True)

        q_new = pmf @ cond
        if np.max(np.abs(q_new - q_hat)) < tol:
            q_hat = q_new
            break
        q_hat = q_new

    expected_d = np.sum(pmf[:, None] * cond * distortion_matrix)
    joint = pmf[:, None] * cond
    marg = joint.sum(axis=0)
    mi = 0.0
    for i in range(n):
        for j in range(m):
            if joint[i, j] > eps:
                mi += joint[i, j] * np.log2(
                    joint[i, j] / (pmf[i] * marg[j] + eps)
                )

    return RichResult(payload={"rate": max(mi, 0.0), "distortion": expected_d, "optimal_mapping": cond})
