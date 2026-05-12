"""De Rham cohomology of the n-torus."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def torus_cohomology(dim: int = 2) -> DescriptiveResult:
    r"""Compute de Rham cohomology groups of the n-torus T^n.

    By Poincare duality and the Kuenneth theorem:
    :math:`H^k_{dR}(T^n) \\cong \\mathbb{R}^{\\binom{n}{k}}`

    :param dim: Dimension of the torus (n >= 1).
    :return: DescriptiveResult with cohomology groups and Betti numbers.
    """
    if dim < 1:
        raise ValueError(f"Dimension must be >= 1, got {dim}.")
    groups = {}
    betti = []
    for k in range(dim + 1):
        rank = math.comb(dim, k)
        betti.append(rank)
        groups[f"H^{k}"] = f"R^{rank}" if rank > 1 else "R" if rank == 1 else "0"
    return DescriptiveResult(
        name="torus_cohomology",
        value=float(sum(betti)),
        extra={
            "cohomology_groups": groups,
            "betti_numbers": betti,
            "total_betti": sum(betti),
            "dim": dim,
        },
    )


def cheatsheet() -> str:
    return "torus_cohomology(dim) -> de Rham cohomology of T^n"
