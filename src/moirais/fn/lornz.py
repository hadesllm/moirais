# moirais.fn — function file (hadesllm/moirais)
"""Lorenz curve coordinates. 'Knowledge itself is power. -- Bacon'"""
from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lorenz_curve(
    incomes: np.ndarray,
) -> DescriptiveResult:
    """
    Compute Lorenz curve coordinates for income distribution.

    The Lorenz curve plots cumulative share of population (x-axis)
    against cumulative share of income (y-axis).

    :param incomes: Array of non-negative income values.
    :return: DescriptiveResult with x/y coordinates and area under curve.
    :raises ValueError: If any income is negative or all are zero.

    References
    ----------
    Lorenz, M. O. (1905). Methods of measuring the concentration of
    wealth. *Publications of the American Statistical Association*,
    9(70), 209-219.
    """
    y = np.asarray(incomes, dtype=np.float64).ravel()
    if np.any(y < 0):
        raise ValueError("All income values must be non-negative.")
    if np.sum(y) == 0:
        raise ValueError("Total income must be positive.")

    y_sorted = np.sort(y)
    n = len(y_sorted)

    cum_income = np.cumsum(y_sorted)
    cum_pop = np.arange(1, n + 1) / n
    cum_income_share = cum_income / cum_income[-1]

    x_lorenz = np.concatenate([[0.0], cum_pop])
    y_lorenz = np.concatenate([[0.0], cum_income_share])

    area_under = float(np.trapezoid(y_lorenz, x_lorenz))

    return DescriptiveResult(
        name="Lorenz Curve",
        value=area_under,
        extra={
            "x": x_lorenz,
            "y": y_lorenz,
            "area_under_curve": area_under,
            "gini_from_lorenz": float(1.0 - 2.0 * area_under),
            "n": n,
        },
    )


short = lorenz_curve


def cheatsheet() -> str:
    return "lorenz_curve({}) -> Lorenz curve coordinates. 'Never tell me the odds.' -- Han S"
