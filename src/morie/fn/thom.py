"""Homology groups of the n-torus."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def torus_homology(dim: int = 2) -> DescriptiveResult:
    r"""Compute homology groups of the n-dimensional torus T^n.

    By the Kuenneth formula for the product of *n* circles:
    :math:`H_k(T^n; \\mathbb{Z}) \\cong \\mathbb{Z}^{\\binom{n}{k}}`

    :param dim: Dimension of the torus (n >= 1).
    :return: DescriptiveResult with homology groups in *extra*.
    """
    if dim < 1:
        raise ValueError(f"Dimension must be >= 1, got {dim}.")
    groups = {}
    betti = []
    for k in range(dim + 1):
        rank = math.comb(dim, k)
        betti.append(rank)
        groups[f"H_{k}"] = f"Z^{rank}" if rank > 1 else "Z" if rank == 1 else "0"
    euler_char = sum((-1) ** k * b for k, b in enumerate(betti))
    return DescriptiveResult(
        name="torus_homology",
        value=float(euler_char),
        extra={
            "homology_groups": groups,
            "betti_numbers": betti,
            "euler_characteristic": euler_char,
            "dim": dim,
        },
    )


def cheatsheet() -> str:
    return "torus_homology(dim) -> H_k(T^n) via Kuenneth formula"
