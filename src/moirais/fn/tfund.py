"""Fundamental group of the torus."""

from __future__ import annotations

from ._containers import DescriptiveResult


def torus_fundamental_group(dim: int = 2) -> DescriptiveResult:
    """Fundamental group of the n-dimensional torus T^n.

    :math:`\\pi_1(T^n) \\cong \\mathbb{Z}^n`

    The n-torus is the product of *n* circles, so its fundamental group
    is the free abelian group on *n* generators.

    :param dim: Dimension of the torus (n >= 1).
    :return: DescriptiveResult with group presentation in *extra*.
    :raises ValueError: If dim < 1.
    """
    if dim < 1:
        raise ValueError(f"Dimension must be >= 1, got {dim}.")
    generators = [f"a_{i + 1}" for i in range(dim)]
    relations = [f"[{generators[i]}, {generators[j]}] = 1" for i in range(dim) for j in range(i + 1, dim)]
    return DescriptiveResult(
        name="torus_fundamental_group",
        value=dim,
        extra={
            "group": f"Z^{dim}",
            "generators": generators,
            "relations": relations,
            "abelian": True,
            "rank": dim,
        },
    )


def cheatsheet() -> str:
    return "torus_fundamental_group(dim) -> pi_1(T^n) = Z^n"
