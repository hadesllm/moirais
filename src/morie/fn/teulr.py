"""Euler characteristic of a surface."""

from __future__ import annotations

from ._containers import DescriptiveResult


def torus_euler_char(genus: int = 1) -> DescriptiveResult:
    r"""Euler characteristic of a closed orientable surface of genus *g*.

    :math:`\\chi = 2 - 2g`

    For a torus (g=1), chi=0. For a sphere (g=0), chi=2.

    :param genus: Genus of the surface (number of handles, >= 0).
    :return: DescriptiveResult with chi in *extra*.
    :raises ValueError: If genus < 0.
    """
    if genus < 0:
        raise ValueError(f"Genus must be >= 0, got {genus}.")
    chi = 2 - 2 * genus
    return DescriptiveResult(
        name="torus_euler_char",
        value=float(chi),
        extra={"chi": chi, "genus": genus, "orientable": True},
    )


def cheatsheet() -> str:
    return "torus_euler_char(genus) -> chi = 2 - 2g"
