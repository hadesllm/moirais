# moirais.fn — function file (hadesllm/moirais)
"""Jackknife delete-d estimator. 'Maximum effort!' -- Deadpool"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ._containers import DescriptiveResult


def jackknife_delete_d(
    data: np.ndarray | list[float],
    *,
    d: int = 1,
    statistic: str = "mean",
    max_combos: int = 5000,
    seed: int | None = None,
) -> DescriptiveResult:
    """Compute the delete-d jackknife estimate of bias and variance.

    Generalises the standard jackknife by deleting *d* observations at a time.
    For large n choose d, subsamples randomly up to *max_combos*.

    Parameters
    ----------
    data : array-like
        Observed data (1D).
    d : int
        Number of observations to delete per jackknife sample.
    statistic : str
        "mean", "median", or "var".
    max_combos : int
        Maximum subsamples when :math:`\\binom{n}{d}` is too large.
    seed : int or None
        RNG seed for subsampling.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``estimate``, ``bias``, ``se``,
        ``n_replicates``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or len(x) < 3:
        raise ValueError("data must be 1D with at least 3 elements")
    n = len(x)
    if d < 1 or d >= n:
        raise ValueError(f"d must be in [1, {n - 1}]")

    fns = {"mean": np.mean, "median": np.median, "var": lambda a: float(np.var(a, ddof=1))}
    if statistic not in fns:
        raise ValueError(f"Unknown statistic: {statistic}")
    fn = fns[statistic]

    theta_hat = float(fn(x))

    from math import comb

    total = comb(n, d)
    if total <= max_combos:
        idx_iter = combinations(range(n), d)
        n_rep = total
    else:
        rng = np.random.default_rng(seed)
        idx_iter = (tuple(sorted(rng.choice(n, d, replace=False))) for _ in range(max_combos))
        n_rep = max_combos

    jack_vals = []
    for drop_idx in idx_iter:
        mask = np.ones(n, dtype=bool)
        for i in drop_idx:
            mask[i] = False
        jack_vals.append(fn(x[mask]))

    jack_arr = np.array(jack_vals)
    jack_mean = float(jack_arr.mean())
    bias = (n - d) / d * (jack_mean - theta_hat)
    c = (n - d) / (d * n_rep)
    se = float(np.sqrt(c * np.sum((jack_arr - jack_mean) ** 2)))

    return DescriptiveResult(
        name="jackknife_delete_d",
        value={
            "estimate": theta_hat,
            "bias": float(bias),
            "se": se,
            "n_replicates": n_rep,
        },
        extra={"n": n, "d": d, "statistic": statistic},
    )


dpool = jackknife_delete_d


def cheatsheet() -> str:
    return "jackknife_delete_d({}) -> Jackknife delete-d estimator. 'Maximum effort!' -- Deadpool"
