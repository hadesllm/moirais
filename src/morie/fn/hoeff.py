# morie.fn — function file (hadesllm/morie)
"""Hoeffding concentration inequality bound."""

from __future__ import annotations

import numpy as np


def hoeffding_bound(
    n: int,
    t: float,
    *,
    a: float = 0.0,
    b: float = 1.0,
) -> dict:
    r"""
    Hoeffding concentration inequality for bounded random variables.

    For independent random variables :math:`X_1, \ldots, X_n` with
    :math:`X_i \in [a_i, b_i]`, the Hoeffding inequality states:

    .. math::

        P\!\left(\left|\bar{X}_n - E[\bar{X}_n]\right| \ge t\right)
        \le 2\exp\!\left(\frac{-2n^2 t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)

    For identically bounded variables (:math:`a_i = a`, :math:`b_i = b`):

    .. math::

        P\!\left(\left|\bar{X}_n - E[\bar{X}_n]\right| \ge t\right)
        \le 2\exp\!\left(\frac{-2n t^2}{(b - a)^2}\right)

    :param n: Sample size (number of independent variables).
    :param t: Deviation threshold (> 0).
    :param a: Lower bound of each variable. Default 0.
    :param b: Upper bound of each variable. Default 1.
    :return: dict with ``bound`` (tail probability upper bound),
        ``exponent``, ``n``, ``t``, ``range``.
    :raises ValueError: If n < 1, t <= 0, or a >= b.

    References
    ----------
    Hoeffding, W. (1963). Probability inequalities for sums of bounded
        random variables. *JASA*, 58(301), 13--30.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Theorem 5.1. Springer.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}.")
    if a >= b:
        raise ValueError(f"a must be < b, got a={a}, b={b}.")

    range_sq = (b - a) ** 2
    exponent = -2.0 * n * t**2 / range_sq
    bound = 2.0 * np.exp(exponent)
    bound = min(bound, 1.0)

    return {
        "bound": float(bound),
        "exponent": float(exponent),
        "n": n,
        "t": float(t),
        "range": float(b - a),
        "method": "Hoeffding inequality",
    }


hoeff = hoeffding_bound


def cheatsheet() -> str:
    return "hoeffding_bound(n, t) -> Hoeffding concentration inequality bound."
