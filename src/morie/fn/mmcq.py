# morie.fn -- function file (hadesllm/morie)
"""M/M/c multi-server queue steady-state metrics."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def mmc_queue(
    arrival_rate: float,
    service_rate: float,
    c: int = 2,
) -> DescriptiveResult:
    r"""M/M/c multi-server queue steady-state metrics.

    Uses the Erlang-C formula for the probability of queuing:

    .. math::

        C(c, a) = \\frac{\\frac{a^c}{c!} \\cdot \\frac{1}{1 - a/c}}
        {\\sum_{k=0}^{c-1} \\frac{a^k}{k!}
        + \\frac{a^c}{c!} \\cdot \\frac{1}{1 - a/c}}

    where :math:`a = \\lambda / \\mu`.

    Parameters
    ----------
    arrival_rate : float
        Mean arrival rate (lambda > 0).
    service_rate : float
        Per-server service rate (mu > 0).
    c : int, default 2
        Number of parallel servers (c >= 1).

    Returns
    -------
    DescriptiveResult
        ``value`` is rho (per-server utilization).  ``extra`` has
        ``L``, ``Lq``, ``W``, ``Wq``, ``P0``, ``P_queue`` (Erlang C).

    Raises
    ------
    ValueError
        If rates are non-positive, c < 1, or system unstable.

    References
    ----------
    Erlang, A. K. (1917). Solution of some problems in the theory of
    probabilities of significance in automatic telephone exchanges.
    *Elektroteknikeren*, 13.
    """
    if arrival_rate <= 0:
        raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}.")
    if service_rate <= 0:
        raise ValueError(f"service_rate must be > 0, got {service_rate}.")
    if c < 1:
        raise ValueError(f"c must be >= 1, got {c}.")

    a = arrival_rate / service_rate
    rho = a / c
    if rho >= 1:
        raise ValueError(f"System unstable: rho={rho:.4f} >= 1 (lambda={arrival_rate}, mu={service_rate}, c={c}).")

    sum_terms = sum(a**k / math.factorial(k) for k in range(c))
    last_term = (a**c / math.factorial(c)) * (1 / (1 - rho))
    P0 = 1.0 / (sum_terms + last_term)
    P_queue = last_term * P0

    Lq = P_queue * rho / (1 - rho)
    L = Lq + a
    Wq = Lq / arrival_rate
    W = Wq + 1 / service_rate

    return DescriptiveResult(
        name="MMcQueue",
        value=float(rho),
        extra={
            "L": float(L),
            "Lq": float(Lq),
            "W": float(W),
            "Wq": float(Wq),
            "P0": float(P0),
            "P_queue": float(P_queue),
            "c": c,
            "arrival_rate": arrival_rate,
            "service_rate": service_rate,
        },
    )


mmcq = mmc_queue


def cheatsheet() -> str:
    return "mmc_queue({}) -> M/M/c multi-server queue steady-state metrics."
