# morie.fn -- function file (hadesllm/morie)
"""Ergodic decomposition theorem verification."""

__all__ = ["ergdc"]

import numpy as np


def ergdc(
    transition_matrix: np.ndarray,
    *,
    n_steps: int = 1000,
    tol: float = 1e-8,
) -> dict:
    r"""
    Verify ergodic properties of a discrete-time Markov chain.

    Checks whether a Markov chain is ergodic (irreducible + aperiodic)
    and computes the stationary distribution via eigendecomposition.

    An ergodic chain satisfies:

    .. math::

        \\lim_{n \\to \\infty} P^n = \\mathbf{1} \\pi^T

    Parameters
    ----------
    transition_matrix : np.ndarray
        Row-stochastic matrix P, shape (n, n). P[i,j] = Pr(j | i).
    n_steps : int
        Number of steps for power iteration verification.
    tol : float
        Convergence tolerance for stationarity check.

    Returns
    -------
    dict
        'is_ergodic' (bool),
        'is_irreducible' (bool),
        'is_aperiodic' (bool),
        'stationary_dist' (np.ndarray or None),
        'mixing_time' (int, steps to converge within tol),
        'n_communicating_classes' (int),
        'spectral_gap' (float, 1 - |lambda_2|).

    References
    ----------
    Levin, D. A., Peres, Y. & Wilmer, E. L. (2009). Markov Chains and
    Mixing Times. AMS.
    """
    P = np.asarray(transition_matrix, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("transition_matrix must be square.")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows of transition_matrix must sum to 1.")
    if np.any(P < 0):
        raise ValueError("transition_matrix entries must be non-negative.")

    n = P.shape[0]

    reachable = np.eye(n, dtype=bool)
    power = P > 0
    for _ in range(n):
        reachable = reachable | power.astype(bool)
        power = power @ (P > 0)
    is_irreducible = bool(np.all(reachable))

    if is_irreducible:
        n_classes = 1
    else:
        visited = np.zeros(n, dtype=bool)
        n_classes = 0
        for start in range(n):
            if not visited[start]:
                queue = [start]
                visited[start] = True
                while queue:
                    node = queue.pop(0)
                    for neighbor in range(n):
                        if reachable[node, neighbor] and reachable[neighbor, node] and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                n_classes += 1

    from math import gcd
    is_aperiodic = True
    if is_irreducible:
        return_times = []
        for i in range(n):
            pk = np.eye(n)
            for t in range(1, min(n + 2, 100)):
                pk = pk @ P
                if pk[i, i] > tol:
                    return_times.append(t)
                    break
        if return_times:
            d = return_times[0]
            for t in return_times[1:]:
                d = gcd(d, t)
            is_aperiodic = (d == 1)
    else:
        is_aperiodic = False

    is_ergodic = is_irreducible and is_aperiodic

    stationary = None
    mixing_time = n_steps
    spectral_gap = 0.0

    if is_irreducible:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum()
        stationary = pi

        eigs_sorted = np.sort(np.abs(eigenvalues))[::-1]
        if len(eigs_sorted) > 1:
            spectral_gap = float(1.0 - eigs_sorted[1])
        else:
            spectral_gap = 1.0

        if is_ergodic:
            dist = np.ones(n) / n
            for t in range(1, n_steps + 1):
                dist = dist @ P
                if np.max(np.abs(dist - stationary)) < tol:
                    mixing_time = t
                    break

    return {
        "is_ergodic": is_ergodic,
        "is_irreducible": is_irreducible,
        "is_aperiodic": is_aperiodic,
        "stationary_dist": stationary,
        "mixing_time": mixing_time,
        "n_communicating_classes": n_classes,
        "spectral_gap": spectral_gap,
    }
