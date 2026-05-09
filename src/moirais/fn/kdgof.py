# moirais.fn — function file (hadesllm/moirais)
"""Kernel goodness-of-fit test."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kdgof"]


def kdgof(
    data: np.ndarray,
    *,
    cdf_func: str = "normal",
    bw: float | None = None,
    n_grid: int = 512,
    n_boot: int = 999,
    seed: int = 42,
) -> dict:
    r"""
    Kernel-based goodness-of-fit test (integrated squared difference).

    Tests :math:`H_0: f = f_0` via the test statistic:

    .. math::

        T_n = n h^{1/2} \int [\hat{f}_h(x) - f_0(x)]^2\,dx

    with a bootstrap p-value.

    Parameters
    ----------
    data : np.ndarray
        1-d observations.
    cdf_func : str
        ``'normal'`` (fitted) or ``'uniform'``.
    bw : float or None
        Bandwidth. Silverman's rule if None.
    n_grid : int
        Integration grid size.
    n_boot : int
        Bootstrap replicates.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``bw``.

    References
    ----------
    Fan, Y. (1994). Testing the goodness of fit of a parametric density
        function by kernel method. *Econometric Theory*, 10(2), 316-356.
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

    lo = data.min() - 4 * bw
    hi = data.max() + 4 * bw
    x_grid = np.linspace(lo, hi, n_grid)
    dx = x_grid[1] - x_grid[0]

    def _kde(sample, grid, h):
        u = (grid[:, None] - sample[None, :]) / h
        return np.exp(-0.5 * u ** 2).mean(axis=1) / (h * np.sqrt(2 * np.pi))

    def _stat(sample):
        f_hat = _kde(sample, x_grid, bw)
        if cdf_func == "normal":
            m, s2 = np.mean(sample), max(np.std(sample, ddof=1), 1e-10)
            f0 = norm.pdf(x_grid, loc=m, scale=s2)
        elif cdf_func == "uniform":
            a, b = sample.min(), sample.max()
            f0 = uniform.pdf(x_grid, loc=a, scale=max(b - a, 1e-10))
        else:
            raise ValueError(f"Unknown cdf_func '{cdf_func}'.")
        return float(np.sum((f_hat - f0) ** 2) * dx * n * np.sqrt(bw))

    stat = _stat(data)

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    if cdf_func == "normal":
        mu, sig = np.mean(data), max(np.std(data, ddof=1), 1e-10)
    else:
        a, b = data.min(), data.max()

    for i in range(n_boot):
        if cdf_func == "normal":
            bs = rng.normal(mu, sig, n)
        else:
            bs = rng.uniform(a, b, n)
        boot_stats[i] = _stat(bs)

    p_value = float(np.mean(boot_stats >= stat))

    return RichResult(payload={"statistic": stat, "p_value": p_value, "bw": bw})


def cheatsheet() -> str:
    return "kdgof({data}) -> Kernel goodness-of-fit test (ISE statistic)."
