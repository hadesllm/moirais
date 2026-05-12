"""Bosonic string partition function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def string_partition(
    tau: complex = 0.5j,
    d: int = 26,
    n_terms: int = 50,
) -> DescriptiveResult:
    r"""Compute the bosonic string one-loop partition function.

    .. math::

        Z(\\tau) = \\frac{1}{(\\text{Im}\\,\\tau)^{d/2}\\,|\\eta(\\tau)|^{2d}}

    where :math:`\\eta(\\tau) = q^{1/24} \\prod_{n=1}^\\infty (1 - q^n)`,
    :math:`q = e^{2\\pi i \\tau}`.

    :param tau: Modular parameter (Im(tau) > 0).
    :param d: Number of spacetime dimensions.
    :param n_terms: Terms in the Dedekind eta product.
    :return: DescriptiveResult with partition function value.
    """
    tau = complex(tau)
    if tau.imag <= 0:
        raise ValueError(f"Im(tau) must be > 0, got {tau.imag}.")
    q = np.exp(2j * np.pi * tau)
    eta = q ** (1.0 / 24.0)
    for n in range(1, n_terms + 1):
        eta *= 1 - q**n
    eta_abs = abs(eta)
    if eta_abs == 0:
        Z = float("inf")
    else:
        Z = 1.0 / (tau.imag ** (d / 2.0) * eta_abs ** (2 * d))
    return DescriptiveResult(
        name="string_partition",
        value=float(Z) if np.isfinite(Z) else Z,
        extra={
            "tau": tau,
            "d": d,
            "eta_abs": float(eta_abs),
            "q_abs": float(abs(q)),
            "im_tau": float(tau.imag),
        },
    )


def cheatsheet() -> str:
    return "string_partition(tau, d) -> bosonic string partition Z(tau)"


strpf = string_partition
