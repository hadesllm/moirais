# moirais.fn — function file (hadesllm/moirais)
"""Discount future costs/effects."""

import numpy as np

from ._containers import DescriptiveResult


def discount_rate(
    values: list | np.ndarray,
    years: list | np.ndarray,
    rate: float = 0.03,
) -> DescriptiveResult:
    """Discount future values to present value.

    .. math::

        PV = \\sum_t \\frac{V_t}{(1 + r)^t}

    Parameters
    ----------
    values : array-like
        Future values.
    years : array-like
        Year offsets from present (0, 1, 2, ...).
    rate : float
        Annual discount rate (default 3%).

    Returns
    -------
    DescriptiveResult
    """
    v = np.asarray(values, dtype=float)
    y = np.asarray(years, dtype=float)
    if len(v) != len(y):
        raise ValueError("values and years must match")

    discounted = v / (1 + rate) ** y
    pv = float(np.sum(discounted))

    return DescriptiveResult(
        name="discounted_value",
        value=pv,
        extra={"undiscounted_total": float(np.sum(v)), "discount_rate": rate, "n_periods": len(v)},
    )


hedsc = discount_rate


def cheatsheet() -> str:
    return "discount_rate({}) -> Discount future costs/effects."
