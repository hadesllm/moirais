"""Approximate Shortest Vector Problem via LLL."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def svp_approx(basis: np.ndarray, delta: float = 0.75) -> DescriptiveResult:
    """Approximate SVP by returning the shortest vector from LLL reduction.

    :param basis: n x m matrix (rows are basis vectors).
    :param delta: LLL parameter.
    :return: DescriptiveResult with approximate shortest vector.
    """
    from moirais.crypto._lattice_core import svp_approx as _svp

    B = np.asarray(basis, dtype=np.float64)
    sv = _svp(B, delta=delta)
    norm = float(np.linalg.norm(sv))
    return DescriptiveResult(
        name="svp_approx",
        value=norm,
        extra={
            "shortest_vector": sv,
            "norm": norm,
            "dimension": B.shape[0],
            "delta": delta,
        },
    )


svpap = svp_approx


def cheatsheet() -> str:
    return "svp_approx({}) -> Approximate Shortest Vector Problem via LLL."
