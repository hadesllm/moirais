# morie.fn — function file (hadesllm/morie)
"""M/M/1 queue steady-state metrics."""

from __future__ import annotations

from ._containers import DescriptiveResult


def mm1_queue(
    arrival_rate: float,
    service_rate: float,
) -> DescriptiveResult:
    """M/M/1 single-server queue steady-state metrics.

    Traffic intensity :math:`\\rho = \\lambda / \\mu`.  The system is
    stable when :math:`\\rho < 1`.

    .. math::

        L = \\frac{\\rho}{1 - \\rho}, \\quad
        W = \\frac{1}{\\mu - \\lambda}, \\quad
        L_q = \\frac{\\rho^2}{1 - \\rho}, \\quad
        W_q = \\frac{\\rho}{\\mu - \\lambda}

    Parameters
    ----------
    arrival_rate : float
        Mean arrival rate (lambda > 0).
    service_rate : float
        Mean service rate (mu > 0, mu > lambda for stability).

    Returns
    -------
    DescriptiveResult
        ``value`` is the traffic intensity rho.  ``extra`` has ``L``
        (mean customers in system), ``Lq`` (in queue), ``W`` (mean
        time in system), ``Wq`` (mean wait in queue), ``P0`` (idle
        probability).

    Raises
    ------
    ValueError
        If rates are non-positive or system is unstable.

    References
    ----------
    Gross, D., Shortle, J. F., Thompson, J. M., & Harris, C. M. (2008).
    *Fundamentals of Queueing Theory* (4th ed.). Wiley.
    """
    if arrival_rate <= 0:
        raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}.")
    if service_rate <= 0:
        raise ValueError(f"service_rate must be > 0, got {service_rate}.")

    rho = arrival_rate / service_rate
    if rho >= 1:
        raise ValueError(f"System unstable: rho={rho:.4f} >= 1 (lambda={arrival_rate}, mu={service_rate}).")

    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1 / (service_rate - arrival_rate)
    Wq = rho / (service_rate - arrival_rate)
    P0 = 1 - rho

    return DescriptiveResult(
        name="MM1Queue",
        value=float(rho),
        extra={
            "L": float(L),
            "Lq": float(Lq),
            "W": float(W),
            "Wq": float(Wq),
            "P0": float(P0),
            "arrival_rate": arrival_rate,
            "service_rate": service_rate,
        },
    )


mm1q = mm1_queue


def cheatsheet() -> str:
    return "mm1_queue({}) -> M/M/1 queue steady-state metrics."
