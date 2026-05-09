# moirais.fn — function file (hadesllm/moirais)
"""Holographic entanglement entropy (Ryu-Takayanagi)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def holographic_entropy(
    area: float = 1.0,
    G_N: float = 1.0,
) -> DescriptiveResult:
    """Compute holographic entanglement entropy via the Ryu-Takayanagi formula.

    .. math::

        S_A = \\frac{\\text{Area}(\\gamma_A)}{4 G_N}

    where :math:`\\gamma_A` is the minimal surface in the bulk homologous
    to boundary region A.

    :param area: Area of the minimal (RT) surface. Must be >= 0.
    :param G_N: Newton's constant in the bulk. Must be > 0.
    :return: DescriptiveResult with holographic entropy.
    """
    if area < 0:
        raise ValueError(f"Area must be >= 0, got {area}.")
    if G_N <= 0:
        raise ValueError(f"G_N must be > 0, got {G_N}.")
    entropy = area / (4.0 * G_N)
    return DescriptiveResult(
        name="holographic_entropy",
        value=float(entropy),
        extra={
            "entropy": float(entropy),
            "area": area,
            "G_N": G_N,
        },
    )


def cheatsheet() -> str:
    return "holographic_entropy(area, G_N) -> Ryu-Takayanagi S = Area/(4G_N)"


holog = holographic_entropy
