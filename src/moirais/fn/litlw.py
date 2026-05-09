# moirais.fn — function file (hadesllm/moirais)
"""Little's law: L = lambda * W."""

from __future__ import annotations

from ._containers import DescriptiveResult


def littles_law(
    arrival_rate: float,
    avg_wait: float,
) -> DescriptiveResult:
    """Apply Little's law to compute mean number in system.

    .. math::

        L = \\lambda \\, W

    This fundamental result holds for any stationary queueing system
    regardless of arrival or service distributions.

    Parameters
    ----------
    arrival_rate : float
        Long-run average arrival rate (lambda > 0).
    avg_wait : float
        Long-run average time a customer spends in the system
        (W >= 0).

    Returns
    -------
    DescriptiveResult
        ``value`` is *L* (mean number in system).  ``extra`` has
        ``arrival_rate`` and ``avg_wait``.

    Raises
    ------
    ValueError
        If arrival_rate <= 0 or avg_wait < 0.

    References
    ----------
    Little, J. D. C. (1961). A proof for the queuing formula:
    L = lambda W. *Operations Research*, 9(3), 383--387.

    Little, J. D. C., & Graves, S. C. (2008). Little's law.
    In D. Chhajed & T. J. Lowe (Eds.), *Building Intuition*
    (pp. 81--100). Springer.
    """
    if arrival_rate <= 0:
        raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}.")
    if avg_wait < 0:
        raise ValueError(f"avg_wait must be >= 0, got {avg_wait}.")

    L = arrival_rate * avg_wait

    return DescriptiveResult(
        name="LittlesLaw",
        value=float(L),
        extra={
            "arrival_rate": arrival_rate,
            "avg_wait": avg_wait,
        },
    )


litlw = littles_law


def cheatsheet() -> str:
    return "littles_law({}) -> Little's law: L = lambda * W."
