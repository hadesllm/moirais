# morie.fn — function file (hadesllm/morie)
"""Kernel-smoothed Kolmogorov-Smirnov test."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kscvs"]


def kscvs(data: np.ndarray, cdf=None, *, cdf_func: str = "normal", bw: float | None = None, n_grid: int = 1024, n_boot: int = 999, seed: int = 42) -> dict:
    r"""
    Kolmogorov-Smirnov test using a kernel-smoothed empirical CDF.

    Computes the KS statistic between the kernel CDF estimator and a
    reference distribution, with a bootstrap p-value:

    .. math::

        D_n = \sup_x |\hat{F}_h(x) - F_0(x)|

    Parameters
    ----------
    data : np.ndarray
        1-d observations.
    cdf_func : str
        Reference distribution: ``'normal'`` (fitted) or ``'uniform'``.
    bw : float or None
        Bandwidth for the kernel CDF. Silverman's rule if None.
    n_grid : int
        CDF evaluation grid size.
    n_boot : int
        Number of bootstrap replicates for the p-value.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``bw``.

    References
    ----------
    Li, Q. & Racine, J. S. (2007). *Nonparametric Econometrics*.
        Princeton University Press. Chapter 12.
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

    x_grid = np.linspace(data.min() - 4 * bw, data.max() + 4 * bw, n_grid)

    def _kernel_cdf(sample, grid, h):
        u = (grid[:, None] - sample[None, :]) / h
        return norm.cdf(u).mean(axis=1)

    if cdf_func == "normal":
        mu, sig = np.mean(data), np.std(data, ddof=1)
        ref_cdf = norm.cdf(x_grid, loc=mu, scale=max(sig, 1e-10))
    elif cdf_func == "uniform":
        a, b = data.min(), data.max()
        ref_cdf = uniform.cdf(x_grid, loc=a, scale=max(b - a, 1e-10))
    else:
        raise ValueError(f"Unknown cdf_func '{cdf_func}'.")

    kcdf = _kernel_cdf(data, x_grid, bw)
    stat = float(np.max(np.abs(kcdf - ref_cdf)))

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    for b_idx in range(n_boot):
        if cdf_func == "normal":
            boot_sample = rng.normal(mu, max(sig, 1e-10), n)
        else:
            boot_sample = rng.uniform(a, b, n)
        bkcdf = _kernel_cdf(boot_sample, x_grid, bw)
        if cdf_func == "normal":
            bmu, bsig = np.mean(boot_sample), np.std(boot_sample, ddof=1)
            bref = norm.cdf(x_grid, loc=bmu, scale=max(bsig, 1e-10))
        else:
            bref = uniform.cdf(x_grid, loc=boot_sample.min(),
                               scale=max(boot_sample.max() - boot_sample.min(), 1e-10))
        boot_stats[b_idx] = np.max(np.abs(bkcdf - bref))

    p_value = float(np.mean(boot_stats >= stat))

    return RichResult(payload={"statistic": stat, "p_value": p_value, "bw": bw})


def cheatsheet() -> str:
    return "kscvs({data}) -> KS test with kernel-smoothed CDF."
