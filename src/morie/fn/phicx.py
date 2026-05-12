# morie.fn -- function file (hadesllm/morie)
"""Phi complexity (integrated information)."""

import numpy as np

from ._containers import ESRes


def phi_complexity(cov_matrix, partition: list[list[int]] | None = None, **kwargs) -> ESRes:
    r"""
    Compute phi (integrated information) for Gaussian variables.

    Phi measures how much a system is more than the sum of its parts.
    Uses the minimum information partition (MIP) heuristic.

    .. math::

        \\Phi = I(X) - \\sum_k I(X_k)

    where {X_k} is a bipartition of the system.

    :param cov_matrix: (p, p) covariance matrix.
    :param partition: Optional bipartition as [[idx1,...], [idx2,...]].
        If None, uses the split that minimises phi (MIP).
    :return: ESRes with phi value.

    References
    ----------
    Tononi G (2004). An information integration theory of consciousness.
    BMC Neuroscience, 5, 42.
    """
    C = np.asarray(cov_matrix, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")
    p = C.shape[0]
    if p < 2:
        raise ValueError("Need at least 2 variables.")

    def _integration(cov):
        variances = np.diag(cov)
        if np.any(variances <= 0):
            return 0.0
        s, ld = np.linalg.slogdet(cov)
        if s <= 0:
            return 0.0
        return 0.5 * float(np.sum(np.log(variances))) - 0.5 * ld

    total_integ = _integration(C)

    if partition is not None:
        parts = partition
    else:
        best_phi = float("inf")
        best_part = [[0], list(range(1, p))]
        for i in range(1, p):
            part_a = [i]
            part_b = [j for j in range(p) if j != i]
            sum_integ = _integration(C[np.ix_(part_a, part_a)]) + _integration(C[np.ix_(part_b, part_b)])
            phi_candidate = total_integ - sum_integ
            if phi_candidate < best_phi:
                best_phi = phi_candidate
                best_part = [part_a, part_b]
        parts = best_part

    sum_parts_integ = sum(_integration(C[np.ix_(part, part)]) for part in parts)
    phi = total_integ - sum_parts_integ

    return ESRes(
        measure="phi_complexity",
        estimate=float(phi),
        n=p,
        extra={"total_integration": total_integ, "partition": parts, "n_variables": p},
    )


phicx = phi_complexity


def cheatsheet() -> str:
    return "phi_complexity(cov_matrix) -> Integrated information phi."
