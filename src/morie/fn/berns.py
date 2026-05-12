# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bernstein concentration inequality."""

from __future__ import annotations

import numpy as np


def bernstein_bound(
    n: int,
    t: float,
    *,
    sigma_sq: float = 0.25,
    M: float = 1.0,
) -> dict:
    r"""
    Bernstein concentration inequality for bounded random variables.

    For independent zero-mean variables :math:`X_i` with
    :math:`|X_i| \le M` and :math:`\sum \mathrm{Var}(X_i) \le v`:

    .. math::

        P\!\left(\sum_{i=1}^n X_i \ge t\right)
        \le \exp\!\left(\frac{-t^2/2}{v + Mt/3}\right)

    For identically distributed variables with variance :math:`\sigma^2`,
    :math:`v = n\sigma^2`.

    :param n: Sample size.
    :param t: Deviation threshold (> 0).
    :param sigma_sq: Per-variable variance bound. Default 0.25.
    :param M: Almost-sure bound on |X_i|. Default 1.0.
    :return: dict with ``bound``, ``exponent``, ``variance_term``,
        ``linear_term``.
    :raises ValueError: If n < 1, t <= 0, sigma_sq <= 0, or M <= 0.

    References
    ----------
    Bernstein, S.N. (1924). On a modification of Chebyshev's inequality
        and on the error of Laplace's formula.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Sec. 5.1. Springer.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}.")
    if sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be > 0, got {sigma_sq}.")
    if M <= 0:
        raise ValueError(f"M must be > 0, got {M}.")

    v = n * sigma_sq
    exponent = -(t**2 / 2.0) / (v + M * t / 3.0)
    bound = min(float(np.exp(exponent)), 1.0)

    return {
        "bound": bound,
        "exponent": float(exponent),
        "variance_term": float(v),
        "linear_term": float(M * t / 3.0),
        "n": n,
        "t": float(t),
        "method": "Bernstein inequality",
    }


berns = bernstein_bound


def cheatsheet() -> str:
    return "bernstein_bound(n, t) -> Bernstein concentration inequality."
