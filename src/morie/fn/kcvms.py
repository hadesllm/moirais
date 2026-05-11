# morie.fn — function file (hadesllm/morie)
"""Cramer-von Mises test with kernel smoothing."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kcvms"]


def kcvms(data: np.ndarray, cdf=None, *, cdf_func: str = "normal", bw: float | None = None, n_grid: int = 1024, n_boot: int = 999, seed: int = 42) -> dict:
    r"""
    Cramer-von Mises test using a kernel-smoothed CDF.

    .. math::

        W^2 = n \int [\hat{F}_h(x) - F_0(x)]^2\, dF_0(x)

    approximated on a fine grid with bootstrap p-value.

    Parameters
    ----------
    data : np.ndarray
        1-d observations.
    cdf_func : str
        ``'normal'`` (fitted) or ``'uniform'``.
    bw : float or None
        Bandwidth. Silverman's rule if None.
    n_grid : int
        Grid size.
    n_boot : int
        Bootstrap replicates for p-value.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``bw``.

    References
    ----------
    Anderson, T. W. (1962). On the distribution of the two-sample
        Cramer-von Mises criterion. *Annals of Mathematical Statistics*,
        33(3), 1148-1159.
    """
    from scipy.stats import norm, uniform

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations.")

    sigma = np.std(data, ddof=1)
    iqr_val = np.subtract(*np.percentile(data, [75, 25]))
    s = min(sigma, iqr_val / 1.349) if iqr_val > 0 else sigma
    if bw is None:
        bw = max(0.9 * s * n ** (-0.2), 1e-10)

    def _kernel_cdf_at(sample, pts, h):
        u = (pts[:, None] - sample[None, :]) / h
        return norm.cdf(u).mean(axis=1)

    if cdf_func == "normal":
        mu, sig = np.mean(data), np.std(data, ddof=1)
        sig = max(sig, 1e-10)
    elif cdf_func == "uniform":
        a, b = data.min(), data.max()
    else:
        raise ValueError(f"Unknown cdf_func '{cdf_func}'.")

    sorted_data = np.sort(data)

    def _cvm_stat(sample):
        pts = np.sort(sample)
        kcdf = _kernel_cdf_at(sample, pts, bw)
        if cdf_func == "normal":
            m, s2 = np.mean(sample), max(np.std(sample, ddof=1), 1e-10)
            ref = norm.cdf(pts, loc=m, scale=s2)
        else:
            lo, hi = sample.min(), sample.max()
            ref = uniform.cdf(pts, loc=lo, scale=max(hi - lo, 1e-10))
        return float(np.mean((kcdf - ref) ** 2))

    stat = _cvm_stat(data)

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        if cdf_func == "normal":
            bs = rng.normal(mu, sig, n)
        else:
            bs = rng.uniform(a, b, n)
        boot_stats[i] = _cvm_stat(bs)

    p_value = float(np.mean(boot_stats >= stat))

    return RichResult(payload={"statistic": stat, "p_value": p_value, "bw": bw})


def cheatsheet() -> str:
    return "kcvms({data}) -> Cramer-von Mises test with kernel smoothing."
