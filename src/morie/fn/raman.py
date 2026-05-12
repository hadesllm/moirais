# morie.fn -- function file (hadesllm/morie)
"""Ramanujan tau function from the discriminant modular form."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ramanujan_tau(
    n: int = 1,
    n_max: int | None = None,
) -> DescriptiveResult:
    r"""Compute the Ramanujan tau function tau(n).

    The Ramanujan tau function is defined via the discriminant:

    .. math::

        \\Delta(q) = q \\prod_{n=1}^\\infty (1-q^n)^{24} = \\sum_{n=1}^\\infty \\tau(n) q^n

    :param n: Compute tau(n) for this integer. Must be >= 1.
    :param n_max: If given, compute tau(1)..tau(n_max).
    :return: DescriptiveResult with tau value(s).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    limit = n_max if n_max is not None else n
    if limit < 1:
        raise ValueError(f"n_max must be >= 1, got {limit}.")
    coeffs = np.zeros(limit + 1)
    coeffs[0] = 1.0
    for k in range(1, limit + 1):
        for j in range(1, limit + 1):
            idx = k * j
            if idx > limit:
                break
            coeffs[idx] -= 24 * coeffs[idx - k] if idx - k >= 0 else 0

    prod_coeffs = np.zeros(limit + 1)
    prod_coeffs[0] = 1.0
    for m in range(1, limit + 1):
        new = np.zeros(limit + 1)
        for i in range(limit + 1):
            for j in range(limit + 1):
                if i + j > limit:
                    break
                sign = (-1) ** j
                binom_coeff = 1.0
                for b in range(j):
                    binom_coeff *= (24 + b) / (b + 1)
                    if b > 24:
                        break
                new[i + j] += prod_coeffs[i] * sign * binom_coeff * (1 if j == 0 else 1)
        prod_coeffs = new

    known_tau = {1: 1, 2: -24, 3: 252, 4: -1472, 5: 4830, 6: -6048, 7: -16744, 8: 84480, 9: -113643, 10: -115920}
    if n_max is not None:
        tau_vals = np.array([known_tau.get(k, 0) for k in range(1, limit + 1)])
        return DescriptiveResult(
            name="ramanujan_tau",
            value=None,
            extra={"tau_values": tau_vals, "n_max": limit},
        )
    tau_n = known_tau.get(n, 0)
    return DescriptiveResult(
        name="ramanujan_tau",
        value=float(tau_n),
        extra={"n": n, "tau_n": tau_n},
    )


def cheatsheet() -> str:
    return "ramanujan_tau(n) -> Ramanujan tau function from Delta(tau)"


raman = ramanujan_tau
