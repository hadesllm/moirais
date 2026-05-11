# morie.fn — function file (hadesllm/morie)
"""Real knowledge is to know the extent of one's ignorance. — Confucius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gini_coefficient(
    incomes: np.ndarray,
) -> DescriptiveResult:
    """
    Compute the Gini coefficient for income inequality.

    .. math::

        G = \\frac{\\sum_{i=1}^{n} \\sum_{j=1}^{n} |y_i - y_j|}
            {2 n^2 \\bar{y}}

    Uses the efficient sorted-array formula:

    .. math::

        G = \\frac{2 \\sum_{i=1}^{n} i \\cdot y_{(i)}}{n \\sum_{i=1}^{n} y_{(i)}}
            - \\frac{n+1}{n}

    :param incomes: Array of non-negative income values.
    :return: DescriptiveResult with Gini coefficient in [0, 1].
    :raises ValueError: If any income is negative or all are zero.

    References
    ----------
    Gini, C. (1912). Variabilita e mutabilita. *Studi Economico-
    Giuridici della R. Universita di Cagliari*, 3, 3-159.
    """
    y = np.asarray(incomes, dtype=np.float64).ravel()
    if np.any(y < 0):
        raise ValueError("All income values must be non-negative.")
    if np.sum(y) == 0:
        raise ValueError("Total income must be positive.")

    y_sorted = np.sort(y)
    n = len(y_sorted)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * y_sorted)) / (n * np.sum(y_sorted)) - (n + 1) / n)

    return DescriptiveResult(
        name="Gini Coefficient",
        value=gini,
        extra={
            "n": n,
            "mean_income": float(np.mean(y)),
            "median_income": float(np.median(y)),
        },
    )


short = gini_coefficient


def cheatsheet() -> str:
    return "gini_coefficient({}) -> Gini inequality coefficient. 'In my experience there is no s"
