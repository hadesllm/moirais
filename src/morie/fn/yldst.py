"""Years lived with disability (YLD)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def years_lived_with_disability(
    cases: list[int] | np.ndarray,
    durations: list[float] | np.ndarray,
    weights: list[float] | np.ndarray,
    discount_rate: float = 0.03,
) -> ESRes:
    r"""Compute years lived with disability (YLD).

    .. math::

        YLD = \\sum_i N_i \\cdot DW_i \\cdot D_i

    where :math:`N_i` is incident cases, :math:`DW_i` is disability
    weight, and :math:`D_i` is average duration.

    Parameters
    ----------
    cases : array-like of int
        Incident cases per condition.
    durations : array-like of float
        Average duration of disability (years) per condition.
    weights : array-like of float
        Disability weights (0-1) per condition.
    discount_rate : float, default 0.03
        Discount rate (0 = no discounting).

    Returns
    -------
    ESRes

    References
    ----------
    Murray, C. J. L. & Lopez, A. D. (1996). *The Global Burden of
    Disease*. Harvard University Press.
    """
    n = np.asarray(cases, dtype=float)
    dur = np.asarray(durations, dtype=float)
    dw = np.asarray(weights, dtype=float)

    if len(n) != len(dur) or len(n) != len(dw):
        raise ValueError("All arrays must have the same length")
    if np.any(dw < 0) or np.any(dw > 1):
        raise ValueError("Disability weights must be in [0, 1]")

    r = discount_rate
    total_yld = 0.0
    for i in range(len(n)):
        if r > 0 and dur[i] > 0:
            disc_dur = (1 - np.exp(-r * dur[i])) / r
        else:
            disc_dur = dur[i]
        total_yld += n[i] * dw[i] * disc_dur

    return ESRes(
        measure="YLD",
        estimate=float(total_yld),
        n=int(np.sum(n)),
        extra={"discount_rate": r, "conditions": len(n)},
    )


yldst = years_lived_with_disability


def cheatsheet() -> str:
    return "years_lived_with_disability({}) -> Years lived with disability (YLD)."
