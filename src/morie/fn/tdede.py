"""Dedekind eta function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dedekind_eta(tau: complex = 1j, terms: int = 50) -> DescriptiveResult:
    r"""Compute the Dedekind eta function eta(tau).

    :math:`\\eta(\\tau) = q^{1/24}\\prod_{n=1}^{N}(1 - q^n)`

    where :math:`q = e^{2\\pi i \\tau}`.

    :param tau: Modular parameter with Im(tau) > 0.
    :param terms: Number of product terms.
    :return: DescriptiveResult with eta value.
    """
    if np.imag(tau) <= 0:
        raise ValueError(f"Im(tau) must be > 0, got {np.imag(tau)}.")
    q = np.exp(2j * np.pi * tau)
    q24 = np.exp(2j * np.pi * tau / 24.0)
    product = 1.0 + 0j
    qn = q
    for _ in range(1, terms + 1):
        product *= 1.0 - qn
        qn *= q
    val = q24 * product
    return DescriptiveResult(
        name="dedekind_eta",
        value=float(np.abs(val)),
        extra={
            "abs": float(np.abs(val)),
            "real": float(np.real(val)),
            "imag": float(np.imag(val)),
            "tau": tau,
            "terms": terms,
        },
    )


def cheatsheet() -> str:
    return "dedekind_eta(tau, terms) -> Dedekind eta function"
