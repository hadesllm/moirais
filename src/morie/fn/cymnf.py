# morie.fn — function file (hadesllm/morie)
"""Calabi-Yau manifold Hodge numbers and Euler characteristic."""

from __future__ import annotations

from ._containers import DescriptiveResult


def calabi_yau_hodge(
    h11: int = 1,
    h21: int = 101,
) -> DescriptiveResult:
    """Compute Calabi-Yau threefold topological invariants from Hodge numbers.

    .. math::

        \\chi = 2(h^{1,1} - h^{2,1})

    The number of complex structure moduli is :math:`h^{2,1}` and
    Kahler moduli is :math:`h^{1,1}`.

    :param h11: Hodge number h^{1,1} (Kahler moduli). Must be >= 0.
    :param h21: Hodge number h^{2,1} (complex structure moduli). Must be >= 0.
    :return: DescriptiveResult with Euler characteristic and moduli counts.
    """
    if h11 < 0 or h21 < 0:
        raise ValueError(f"Hodge numbers must be non-negative, got h11={h11}, h21={h21}.")
    euler = 2 * (h11 - h21)
    total_moduli = h11 + h21
    return DescriptiveResult(
        name="calabi_yau_hodge",
        value=float(euler),
        extra={
            "h11": h11,
            "h21": h21,
            "euler_characteristic": euler,
            "kahler_moduli": h11,
            "complex_structure_moduli": h21,
            "total_moduli": total_moduli,
        },
    )


def cheatsheet() -> str:
    return "calabi_yau_hodge(h11, h21) -> CY Hodge numbers & Euler char"


cymnf = calabi_yau_hodge
