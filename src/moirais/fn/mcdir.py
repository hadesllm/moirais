# moirais.fn — function file (hadesllm/moirais)
"""McDiarmid bounded-differences inequality."""

from __future__ import annotations

import numpy as np


def mcdiarmid_bound(
    n: int,
    t: float,
    *,
    c: float | np.ndarray = 1.0,
) -> dict:
    r"""
    McDiarmid (bounded differences) concentration inequality.

    For independent variables :math:`X_1, \ldots, X_n` and a function
    :math:`f` satisfying the bounded-differences condition:

    .. math::

        \sup_{x_1,\ldots,x_n,x_i'} |f(x_1,\ldots,x_i,\ldots,x_n)
        - f(x_1,\ldots,x_i',\ldots,x_n)| \le c_i

    the inequality states:

    .. math::

        P\!\left(f(X) - E[f(X)] \ge t\right)
        \le \exp\!\left(\frac{-2t^2}{\sum_{i=1}^n c_i^2}\right)

    :param n: Number of independent variables.
    :param t: Deviation threshold (> 0).
    :param c: Bounded-difference constants. Scalar (same for all) or
        array of length n. Default 1.0.
    :return: dict with ``bound``, ``sum_c_sq``, ``exponent``.
    :raises ValueError: If n < 1, t <= 0, or c values non-positive.

    References
    ----------
    McDiarmid, C. (1989). On the method of bounded differences.
        *Surveys in Combinatorics*, London Math. Soc. Lecture Note Ser. 141.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Sec. 5.1. Springer.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}.")

    c_arr = np.broadcast_to(np.asarray(c, dtype=float), (n,))
    if np.any(c_arr <= 0):
        raise ValueError("All bounded-difference constants must be > 0.")

    sum_c_sq = float(np.sum(c_arr**2))
    exponent = -2.0 * t**2 / sum_c_sq
    bound = min(float(np.exp(exponent)), 1.0)

    return {
        "bound": bound,
        "exponent": float(exponent),
        "sum_c_sq": sum_c_sq,
        "n": n,
        "t": float(t),
        "method": "McDiarmid inequality",
    }


mcdir = mcdiarmid_bound


def cheatsheet() -> str:
    return "mcdiarmid_bound(n, t) -> McDiarmid bounded-differences inequality."
