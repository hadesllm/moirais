"""Jacobi theta function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def theta_function(
    z: complex = 0.0,
    tau: complex = 1j,
    terms: int = 30,
) -> DescriptiveResult:
    """Compute the Jacobi theta function theta_3(z, tau).

    :math:`\\theta_3(z, \\tau) = \\sum_{n=-N}^{N} q^{n^2} e^{2\\pi i n z}`

    where :math:`q = e^{\\pi i \\tau}`.

    :param z: Argument.
    :param tau: Nome parameter with Im(tau) > 0.
    :param terms: Summation range [-terms, terms].
    :return: DescriptiveResult with theta value.
    """
    if np.imag(tau) <= 0:
        raise ValueError(f"Im(tau) must be > 0, got {np.imag(tau)}.")
    q = np.exp(1j * np.pi * tau)
    val = 0.0 + 0j
    for n in range(-terms, terms + 1):
        val += q ** (n * n) * np.exp(2j * np.pi * n * z)
    return DescriptiveResult(
        name="theta_function",
        value=float(np.real(val)),
        extra={"real": float(np.real(val)), "imag": float(np.imag(val)), "z": z, "tau": tau, "terms": terms},
    )


def cheatsheet() -> str:
    return "theta_function(z, tau, terms) -> Jacobi theta_3"
