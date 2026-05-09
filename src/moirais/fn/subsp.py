"""Subsampling inference (Politis & Romano)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def subsp(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    b: int | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    r"""
    Subsampling inference for an arbitrary statistic.

    Unlike the bootstrap, subsampling draws subsets of size :math:`b < n`
    **without replacement** and constructs a distribution of the
    renormalized statistic.  Valid under minimal regularity conditions
    (Politis & Romano, 1994) -- in particular, the statistic need not
    be smooth and the data need not be i.i.d.

    The subsampling distribution of :math:`\tau_b(\hat{\theta}_b - \hat{\theta}_n)`
    (where :math:`\tau_b` is the rate, defaulting to :math:`\sqrt{b}`)
    is used to form confidence intervals:

    .. math::

        \hat{\theta}_n \pm \frac{q_{1-\alpha/2}}{\tau_n}

    where :math:`q` are quantiles of the subsampling distribution.

    When the number of possible subsets :math:`\binom{n}{b}` is large,
    a random sample of subsets is drawn (up to 2000).

    :param data: Array of shape ``(n,)`` or ``(n, p)``.
    :param statistic: Callable returning a scalar.
    :param b: Subsample size. Default ``int(n ** 0.7)`` (Politis &
        Romano recommendation).
    :param alpha: Significance level. Default 0.05.
    :param seed: Random seed. Default 42.
    :return: dict with ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``b``, ``n_subsamples``.
    :raises ValueError: If *b* is not in ``[2, n-1]``.

    References
    ----------
    Politis, D. N. & Romano, J. P. (1994). Large sample confidence
        regions based on subsamples under minimal assumptions. *Annals
        of Statistics*, 22(4), 2031--2050.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Section 3.6.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]

    if b is None:
        b = max(2, int(n ** 0.7))
    if b < 2 or b >= n:
        raise ValueError(f"b must be in [2, n-1], got b={b}, n={n}.")
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(data))

    from math import comb

    n_possible = comb(n, b)
    max_sub = 2000
    n_sub = min(n_possible, max_sub)

    tau_b = np.sqrt(b)
    tau_n = np.sqrt(n)

    sub_stats = np.empty(n_sub)
    seen = set()
    i = 0
    while i < n_sub:
        idx = tuple(sorted(rng.choice(n, size=b, replace=False)))
        if idx in seen:
            continue
        seen.add(idx)
        sub_stats[i] = statistic(data[list(idx)])
        i += 1

    centered = tau_b * (sub_stats - theta_hat)

    lo_q = alpha / 2
    hi_q = 1 - alpha / 2
    q_lo = float(np.percentile(centered, lo_q * 100))
    q_hi = float(np.percentile(centered, hi_q * 100))

    ci_lo = theta_hat - q_hi / tau_n
    ci_hi = theta_hat - q_lo / tau_n

    se = float(np.std(sub_stats, ddof=1))

    return {
        "estimate": theta_hat,
        "se": se,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "b": b,
        "n_subsamples": n_sub,
    }


subsp_fn = subsp


def cheatsheet() -> str:
    return "subsp(data, statistic) -> Subsampling inference (Politis & Romano)."
