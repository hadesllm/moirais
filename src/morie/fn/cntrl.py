# morie.fn — function file (hadesllm/morie)
"""Central charge for bosonic and superstring theories."""

from __future__ import annotations

from ._containers import DescriptiveResult


def central_charge(
    d: int = 26,
    supersymmetric: bool = False,
) -> DescriptiveResult:
    """Compute the central charge of a string theory.

    Bosonic string: :math:`c = d` (critical at d=26).
    Superstring: :math:`c = \\frac{3d}{2}` (critical at d=10, c=15).

    :param d: Number of spacetime dimensions.
    :param supersymmetric: If True, use superstring formula.
    :return: DescriptiveResult with central charge.
    """
    if d < 1:
        raise ValueError(f"Dimension must be >= 1, got {d}.")
    c = (3.0 * d / 2.0) if supersymmetric else float(d)
    critical_d = 10 if supersymmetric else 26
    is_critical = d == critical_d
    return DescriptiveResult(
        name="central_charge",
        value=c,
        extra={
            "d": d,
            "supersymmetric": supersymmetric,
            "critical_dimension": critical_d,
            "is_critical": is_critical,
        },
    )


def cheatsheet() -> str:
    return "central_charge(d, supersymmetric) -> central charge c"


cntrl = central_charge
