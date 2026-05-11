# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bekenstein-Hawking entropy S = A/(4 l_P^2)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bekenstein_hawking(
    M: float = 1.0,
    G: float = 1.0,
    c: float = 1.0,
    hbar: float = 1.0,
) -> DescriptiveResult:
    """Compute Bekenstein-Hawking entropy for a Schwarzschild black hole.

    .. math::

        S_{BH} = \\frac{A}{4 \\ell_P^2} = \\frac{4\\pi G M^2}{\\hbar c}

    :param M: Black hole mass.
    :param G: Newton's gravitational constant.
    :param c: Speed of light.
    :param hbar: Reduced Planck constant.
    :return: DescriptiveResult with entropy and horizon area.
    """
    if M <= 0:
        raise ValueError(f"Mass must be > 0, got {M}.")
    r_s = 2 * G * M / c**2
    area = 4 * np.pi * r_s**2
    l_planck = np.sqrt(hbar * G / c**3)
    entropy = area / (4 * l_planck**2)
    hawking_temp = hbar * c**3 / (8 * np.pi * G * M)
    return DescriptiveResult(
        name="bekenstein_hawking",
        value=float(entropy),
        extra={
            "entropy": float(entropy),
            "area": float(area),
            "schwarzschild_radius": float(r_s),
            "hawking_temperature": float(hawking_temp),
            "planck_length": float(l_planck),
            "M": M,
        },
    )


def cheatsheet() -> str:
    return "bekenstein_hawking(M, G, c, hbar) -> BH entropy S = A/(4 l_P^2)"


bkhwk = bekenstein_hawking
