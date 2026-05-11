# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""System availability. 'The garbage will do!' -- Rey"""

from __future__ import annotations

from ._containers import DescriptiveResult


def availability(
    mtbf_val: float,
    mttr: float,
) -> DescriptiveResult:
    r"""Compute steady-state system availability.

    .. math::

        A = \frac{MTBF}{MTBF + MTTR}

    Parameters
    ----------
    mtbf_val : float
        Mean Time Between Failures.
    mttr : float
        Mean Time To Repair.

    Returns
    -------
    DescriptiveResult
        name='Availability', value=availability (0 to 1),
        extra has 'availability', 'unavailability', 'mtbf', 'mttr',
        'nines' (number of nines, e.g. 0.999 = 3 nines).

    References
    ----------
    Ebeling, C.E. (2010). *An Introduction to Reliability and
    Maintainability Engineering* (2nd ed.). Waveland Press. Ch. 6.
    """
    if mtbf_val < 0 or mttr < 0:
        raise ValueError("MTBF and MTTR must be non-negative.")

    total = mtbf_val + mttr
    if total == 0:
        a = 0.0
    else:
        a = mtbf_val / total

    unavail = 1.0 - a

    import math

    nines = -math.log10(max(unavail, 1e-15)) if unavail > 0 else float("inf")

    return DescriptiveResult(
        name="Availability",
        value=float(a),
        extra={
            "availability": float(a),
            "unavailability": float(unavail),
            "mtbf": float(mtbf_val),
            "mttr": float(mttr),
            "nines": float(nines),
        },
    )


avail = availability


def cheatsheet() -> str:
    return "availability({}) -> System availability. 'The garbage will do!' -- Rey"
