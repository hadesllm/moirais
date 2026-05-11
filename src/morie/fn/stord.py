"""Stuart's tau-c for ordinal association. 'We are what they grow beyond.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def stuart_ord(x: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """
    Stuart's tau-c for ordinal association.

    A variant of Kendall's tau designed for rectangular tables where the
    number of rows and columns may differ:

    .. math::

        \\tau_c = \\frac{2m(C - D)}{n^2(m - 1)}

    where *m* = min(r, c), *C* = concordant pairs, *D* = discordant pairs.

    :param x: 1-D ordinal variable.
    :type x: numpy.ndarray
    :param y: 1-D ordinal variable.
    :type y: numpy.ndarray
    :return: DescriptiveResult with tau-c value.
    :rtype: DescriptiveResult

    References
    ----------
    Stuart A. (1953). The Estimation and Comparison of Strengths of
    Association in Contingency Tables. *Biometrika*, 40(1/2), 105-110.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if len(y) != n:
        raise ValueError(f"x ({n}) and y ({len(y)}) must have equal length.")
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    C = 0
    D = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                C += 1
            elif prod < 0:
                D += 1
    r = len(np.unique(x))
    c = len(np.unique(y))
    m = min(r, c)
    tau_c = 2.0 * m * (C - D) / (n**2 * (m - 1)) if m > 1 else 0.0
    return DescriptiveResult(
        name="stuart_tau_c",
        value=float(tau_c),
        extra={
            "tau_c": float(tau_c),
            "concordant": C,
            "discordant": D,
            "n": n,
            "m": m,
        },
    )


stord = stuart_ord


def cheatsheet() -> str:
    return "stuart_ord({}) -> Stuart's tau-c for ordinal association. 'We are what they gr"
