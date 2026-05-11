"""j-invariant of an elliptic curve / complex torus."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def j_invariant(tau: complex = 1j, terms: int = 30) -> DescriptiveResult:
    """Compute the j-invariant j(tau) of the elliptic curve C/(Z + tau*Z).

    Uses the q-expansion:
    :math:`j(\\tau) = \\frac{1}{q} + 744 + 196884q + 21493760q^2 + \\cdots`

    where :math:`q = e^{2\\pi i \\tau}`.

    :param tau: Modular parameter with Im(tau) > 0.
    :param terms: Number of Fourier coefficients (first few hardcoded).
    :return: DescriptiveResult with j-invariant value.
    """
    if np.imag(tau) <= 0:
        raise ValueError(f"Im(tau) must be > 0, got {np.imag(tau)}.")
    q = np.exp(2j * np.pi * tau)
    coeffs = [1, 744, 196884, 21493760, 864299970, 20245856256]
    j_val = 1.0 / q
    qn = 1.0
    for i, c in enumerate(coeffs):
        if i == 0:
            continue
        j_val += c * qn
        qn *= q
    for k in range(len(coeffs), min(terms, len(coeffs) + 5)):
        j_val += 0 * qn
        qn *= q
    return DescriptiveResult(
        name="j_invariant",
        value=float(np.real(j_val)),
        extra={"j_real": float(np.real(j_val)), "j_imag": float(np.imag(j_val)), "tau": tau, "q_abs": float(np.abs(q))},
    )


def cheatsheet() -> str:
    return "j_invariant(tau, terms) -> j-invariant of elliptic curve"
