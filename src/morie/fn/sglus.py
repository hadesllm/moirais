"""LU decomposition for spatial simulation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def lu_decomposition_sim(cov_matrix: np.ndarray) -> SpatialResult:
    r"""You have power over your mind — not outside events. — Marcus Aurelius"""
    from scipy.linalg import lu

    C = np.asarray(cov_matrix, dtype=np.float64)
    P, L, U = lu(C)
    sign, logdet = np.linalg.slogdet(C)

    return SpatialResult(
        name="lu_decomposition_sim",
        statistic=float(logdet),
        p_value=None,
        extra={"L": L, "U": U, "P": P},
    )


sglus = lu_decomposition_sim


def cheatsheet() -> str:
    return "lu_decomposition_sim({}) -> LU decomposition for spatial simulation."
