# morie.fn -- function file (hadesllm/morie)
"""M/G/1 queue via Pollaczek-Khinchine formula."""

from __future__ import annotations

from ._containers import DescriptiveResult


def mg1_queue(
    arrival_rate: float,
    mean_service: float,
    var_service: float,
) -> DescriptiveResult:
    r"""M/G/1 queue metrics via the Pollaczek-Khinchine mean-value formula.

    For a single server with general service-time distribution:

    .. math::

        L_q = \\frac{\\rho^2 + \\lambda^2 \\sigma_s^2}{2(1 - \\rho)}

    where :math:`\\rho = \\lambda / \\mu` and :math:`\\sigma_s^2` is
    the service-time variance.

    Parameters
    ----------
    arrival_rate : float
        Mean arrival rate (lambda > 0).
    mean_service : float
        Mean service time (1/mu > 0).
    var_service : float
        Variance of service time (>= 0).

    Returns
    -------
    DescriptiveResult
        ``value`` is rho.  ``extra`` has ``L``, ``Lq``, ``W``, ``Wq``.

    Raises
    ------
    ValueError
        If parameters invalid or system unstable.

    References
    ----------
    Pollaczek, F. (1930). Uber eine Aufgabe der Wahrscheinlichkeitstheorie.
    *Mathematische Zeitschrift*, 32, 64--100.

    Khinchine, A. Y. (1932). Mathematical theory of a stationary queue.
    *Matematicheskii Sbornik*, 39(4), 73--84.
    """
    if arrival_rate <= 0:
        raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}.")
    if mean_service <= 0:
        raise ValueError(f"mean_service must be > 0, got {mean_service}.")
    if var_service < 0:
        raise ValueError(f"var_service must be >= 0, got {var_service}.")

    mu = 1.0 / mean_service
    rho = arrival_rate / mu
    if rho >= 1:
        raise ValueError(f"System unstable: rho={rho:.4f} >= 1.")

    Lq = (rho**2 + arrival_rate**2 * var_service) / (2 * (1 - rho))
    L = Lq + rho
    Wq = Lq / arrival_rate
    W = Wq + mean_service

    return DescriptiveResult(
        name="MG1Queue",
        value=float(rho),
        extra={
            "L": float(L),
            "Lq": float(Lq),
            "W": float(W),
            "Wq": float(Wq),
            "arrival_rate": arrival_rate,
            "service_rate": float(mu),
            "cv_squared": float(var_service * mu**2),
        },
    )


mg1q = mg1_queue


def cheatsheet() -> str:
    return "mg1_queue({}) -> M/G/1 queue via Pollaczek-Khinchine formula."
