# moirais.fn — function file (hadesllm/moirais)
"""Moving block bootstrap for dependent data."""

from __future__ import annotations

import numpy as np


def moving_block_bootstrap(
    x: np.ndarray,
    *,
    block_size: int | None = None,
    n_boot: int = 500,
    statistic: str = "mean",
    seed: int = 42,
) -> dict:
    r"""
    Moving block bootstrap (MBB) for weakly dependent time series.

    For a stationary process :math:`\{X_1, \ldots, X_n\}`, the MBB
    resamples overlapping blocks of length :math:`l`:

    .. math::

        B_i = (X_i, X_{i+1}, \ldots, X_{i+l-1}), \quad i = 1, \ldots, n-l+1

    Bootstrap samples are formed by concatenating :math:`\lceil n/l \rceil`
    randomly chosen blocks. This preserves the dependence structure
    within blocks.

    If ``block_size`` is None, the default :math:`l = \lfloor n^{1/3} \rfloor`
    is used (optimal rate for smooth statistics under mixing).

    :param x: 1-D array (time series).
    :param block_size: Block length. Default ``n^(1/3)``.
    :param n_boot: Number of bootstrap replications. Default 500.
    :param statistic: Statistic to bootstrap: ``"mean"``, ``"var"``,
        ``"median"``. Default ``"mean"``.
    :param seed: Random seed. Default 42.
    :return: dict with ``estimate`` (point estimate), ``se`` (bootstrap SE),
        ``ci_lower``, ``ci_upper`` (95% percentile CI),
        ``boot_distribution`` (array of bootstrap values).
    :raises ValueError: If x is too short or block_size invalid.

    References
    ----------
    Kunsch, H.R. (1989). The jackknife and the bootstrap for general
        stationary observations. *Ann. Statist.*, 17(3), 1217--1241.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Sec. 2.6 (bootstrap consistency). Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 4:
        raise ValueError(f"Need at least 4 observations, got {n}.")

    if block_size is None:
        block_size = max(1, int(np.floor(n ** (1.0 / 3.0))))
    if block_size < 1 or block_size > n:
        raise ValueError(f"block_size must be in [1, {n}], got {block_size}.")

    stat_fns = {
        "mean": np.mean,
        "var": lambda v: float(np.var(v, ddof=1)) if len(v) > 1 else 0.0,
        "median": np.median,
    }
    if statistic not in stat_fns:
        raise ValueError(f"statistic must be one of {set(stat_fns)}, got '{statistic}'.")

    fn = stat_fns[statistic]
    estimate = float(fn(x))

    n_blocks = n - block_size + 1
    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block_size))

    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n_blocks, size=n_blocks_needed)
        sample = np.concatenate([x[s : s + block_size] for s in starts])[:n]
        boot_vals[b] = fn(sample)

    se = float(np.std(boot_vals, ddof=1))
    ci_lower = float(np.percentile(boot_vals, 2.5))
    ci_upper = float(np.percentile(boot_vals, 97.5))

    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "block_size": block_size,
        "n_boot": n_boot,
        "statistic": statistic,
        "boot_distribution": boot_vals,
        "method": "Moving block bootstrap",
    }


mblbt = moving_block_bootstrap


def cheatsheet() -> str:
    return "moving_block_bootstrap({x}) -> Moving block bootstrap for dependent data."
