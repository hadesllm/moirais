# morie.fn -- function file (hadesllm/morie)
"""Kaluza-Klein mass spectrum from compactification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kaluza_klein_spectrum(
    R: float = 1.0,
    n_max: int = 10,
) -> DescriptiveResult:
    r"""Compute the Kaluza-Klein mass tower for circular compactification.

    .. math::

        m_n = \\frac{n}{R}, \\quad n = 0, 1, 2, \\ldots

    :param R: Compactification radius. Must be > 0.
    :param n_max: Maximum KK mode number.
    :return: DescriptiveResult with mass spectrum array.
    """
    if R <= 0:
        raise ValueError(f"Compactification radius must be > 0, got {R}.")
    if n_max < 0:
        raise ValueError(f"n_max must be >= 0, got {n_max}.")
    modes = np.arange(n_max + 1)
    masses = modes / R
    return DescriptiveResult(
        name="kaluza_klein_spectrum",
        value=float(masses[1]) if n_max >= 1 else 0.0,
        extra={
            "modes": modes,
            "masses": masses,
            "R": R,
            "n_max": n_max,
            "mass_gap": float(1.0 / R),
        },
    )


def cheatsheet() -> str:
    return "kaluza_klein_spectrum(R, n_max) -> KK mass tower m_n = n/R"


klzkl = kaluza_klein_spectrum
