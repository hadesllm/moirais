# moirais.fn — function file (hadesllm/moirais)
"""Maximal inequality bound computation."""

from __future__ import annotations

import numpy as np

__all__ = ["mxinq"]


def mxinq(x: np.ndarray, cdf=None, *, cdf_func: callable | None = None, n_boot: int = 500, alpha: float = 0.05, seed: int | None = None) -> dict:
    r"""
    Compute maximal inequality bounds for the empirical process.

    Estimates :math:`E[\sup_t |\mathbb{G}_n(t)|]` via multiplier bootstrap
    and compares against the DKW bound.

    The maximal inequality states:

    .. math::

        E\!\left[\sup_{f \in \mathcal{F}} |\mathbb{G}_n f|\right]
        \le C \int_0^\infty \sqrt{\log N(\varepsilon, \mathcal{F}, L_2)} \, d\varepsilon

    :param x: 1-D array of observations.
    :param cdf_func: True CDF. Default standard normal.
    :param n_boot: Bootstrap replications. Default 500.
    :param alpha: Significance level for DKW bound. Default 0.05.
    :param seed: Random seed.
    :return: Dict with ``expected_sup``, ``dkw_bound``, ``boot_quantile``,
        ``boot_sups``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 7. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    if cdf_func is None:
        from scipy.stats import norm
        cdf_func = norm.cdf

    rng = np.random.default_rng(seed)
    n = x.size
    x_sorted = np.sort(x)
    ecdf_vals = np.arange(1, n + 1) / n
    true_cdf = cdf_func(x_sorted)

    boot_sups = np.empty(n_boot)
    indicators = (x[:, None] <= x_sorted[None, :]).astype(float)
    for b in range(n_boot):
        w = rng.standard_normal(n)
        gn = w @ (indicators - ecdf_vals[None, :]) / np.sqrt(n)
        boot_sups[b] = np.max(np.abs(gn))

    expected_sup = float(np.mean(boot_sups))
    boot_quantile = float(np.quantile(boot_sups, 1.0 - alpha))
    dkw_bound = float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))

    return {
        "expected_sup": expected_sup,
        "dkw_bound": dkw_bound,
        "boot_quantile": boot_quantile,
        "boot_sups": boot_sups,
        "n": n,
    }


def cheatsheet() -> str:
    return "mxinq({x}) -> Maximal inequality bound for empirical process."
