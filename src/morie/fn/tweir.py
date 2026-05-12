"""Weierstrass p-function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def weierstrass_p(
    z: complex = 0.5 + 0.5j,
    tau: complex = 1j,
    terms: int = 10,
) -> DescriptiveResult:
    r"""Compute the Weierstrass elliptic function P(z; tau).

    Lattice L = Z + tau*Z. Uses the double sum:
    :math:`\\wp(z) = \\frac{1}{z^2} + \\sum_{(m,n)\\neq(0,0)}
    \\left[\\frac{1}{(z - m - n\\tau)^2} - \\frac{1}{(m + n\\tau)^2}\\right]`

    :param z: Point in C (not a lattice point).
    :param tau: Modular parameter (Im(tau) > 0).
    :param terms: Summation range [-terms, terms] for m, n.
    :return: DescriptiveResult with P(z) value.
    """
    if np.imag(tau) <= 0:
        raise ValueError(f"Im(tau) must be > 0, got {np.imag(tau)}.")
    val = 1.0 / z**2
    for m in range(-terms, terms + 1):
        for n in range(-terms, terms + 1):
            if m == 0 and n == 0:
                continue
            omega = m + n * tau
            val += 1.0 / (z - omega) ** 2 - 1.0 / omega**2
    return DescriptiveResult(
        name="weierstrass_p",
        value=float(np.real(val)),
        extra={"real": float(np.real(val)), "imag": float(np.imag(val)), "z": z, "tau": tau, "terms": terms},
    )


def cheatsheet() -> str:
    return "weierstrass_p(z, tau, terms) -> Weierstrass P-function"
