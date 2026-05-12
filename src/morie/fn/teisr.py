"""Eisenstein series E_k."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def eisenstein_series(
    k: int = 4,
    tau: complex = 1j,
    terms: int = 50,
) -> DescriptiveResult:
    r"""Compute the Eisenstein series E_k(tau).

    :math:`E_k(\\tau) = 1 - \\frac{2k}{B_k}\\sum_{n=1}^{N}\\sigma_{k-1}(n)q^n`

    where :math:`q = e^{2\\pi i \\tau}` and sigma_{k-1}(n) is the divisor sum.

    :param k: Weight (even integer >= 4).
    :param tau: Modular parameter with Im(tau) > 0.
    :param terms: Number of Fourier terms.
    :return: DescriptiveResult with E_k value.
    """
    if k < 4 or k % 2 != 0:
        raise ValueError(f"k must be even >= 4, got {k}.")
    if np.imag(tau) <= 0:
        raise ValueError(f"Im(tau) must be > 0, got {np.imag(tau)}.")
    bernoulli = {4: -1.0 / 30, 6: 1.0 / 42, 8: -1.0 / 30, 10: 5.0 / 66, 12: -691.0 / 2730, 14: 7.0 / 6}
    bk = bernoulli.get(k)
    if bk is None:
        raise ValueError(f"Bernoulli number not available for k={k}. Supported: 4,6,8,10,12,14.")
    q = np.exp(2j * np.pi * tau)
    coeff = -2.0 * k / bk
    s = 0.0 + 0j
    qn = q
    for n in range(1, terms + 1):
        sigma = sum(d ** (k - 1) for d in range(1, n + 1) if n % d == 0)
        s += sigma * qn
        qn *= q
    val = 1.0 + coeff * s
    return DescriptiveResult(
        name="eisenstein_series",
        value=float(np.real(val)),
        extra={"real": float(np.real(val)), "imag": float(np.imag(val)), "k": k, "tau": tau},
    )


def cheatsheet() -> str:
    return "eisenstein_series(k, tau, terms) -> Eisenstein series E_k"
