# moirais.fn — function file (hadesllm/moirais)
"""Modular form of weight k."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def modular_form(
    k: int = 4,
    tau: complex = 0.5j,
    terms: int = 20,
) -> DescriptiveResult:
    """Compute the Eisenstein series E_k(tau) as a weight-k modular form.

    .. math::

        E_k(\\tau) = 1 - \\frac{2k}{B_k} \\sum_{n=1}^\\infty \\sigma_{k-1}(n)\\,q^n

    where :math:`q = e^{2\\pi i \\tau}` and :math:`\\sigma_{k-1}` is the
    divisor sum.

    :param k: Weight (must be even >= 4).
    :param tau: Modular parameter (Im(tau) > 0).
    :param terms: Number of q-expansion terms.
    :return: DescriptiveResult with modular form value.
    """
    if k < 4 or k % 2 != 0:
        raise ValueError(f"Weight must be even >= 4, got {k}.")
    tau = complex(tau)
    if tau.imag <= 0:
        raise ValueError(f"Im(tau) must be > 0, got {tau.imag}.")

    bernoulli = {4: -1.0 / 30, 6: 1.0 / 42, 8: -1.0 / 30, 10: 5.0 / 66, 12: -691.0 / 2730, 14: 7.0 / 6}
    B_k = bernoulli.get(k, -1.0 / 30)

    q = np.exp(2j * np.pi * tau)
    result = 1.0 + 0j
    for n in range(1, terms + 1):
        sigma = sum(d ** (k - 1) for d in range(1, n + 1) if n % d == 0)
        result += (-2.0 * k / B_k) * sigma * q**n

    return DescriptiveResult(
        name="modular_form",
        value=float(abs(result)),
        extra={
            "E_k": complex(result),
            "k": k,
            "tau": tau,
            "q_abs": float(abs(q)),
            "terms": terms,
        },
    )


def cheatsheet() -> str:
    return "modular_form(k, tau, terms) -> weight-k Eisenstein series E_k(tau)"


modlr = modular_form
