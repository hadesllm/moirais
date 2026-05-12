# morie.fn -- function file (hadesllm/morie)
"""Exact permutation test for two-sample location."""

import numpy as np

from ._containers import DescriptiveResult


def exact_perm_test(x, y, n_perm=9999, stat="mean"):
    """
    Exact (Monte Carlo) permutation test for two-sample comparison.

    :param x: (n,) first sample.
    :param y: (m,) second sample.
    :param n_perm: Number of permutations.
    :param stat: Test statistic -- 'mean' (difference of means) or 'median'.
    :return: DescriptiveResult with p-value, observed statistic, null distribution.

    References
    ----------
    Good PI (2005). Permutation, Parametric, and Bootstrap Tests of
    Hypotheses. 3rd ed. Springer.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    combined = np.concatenate([x, y])
    nx = len(x)
    n_total = len(combined)
    rng = np.random.default_rng(42)

    if stat == "median":
        obs = float(np.median(x) - np.median(y))
    else:
        obs = float(x.mean() - y.mean())

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        xp, yp = combined[perm[:nx]], combined[perm[nx:]]
        if stat == "median":
            perm_stat = np.median(xp) - np.median(yp)
        else:
            perm_stat = xp.mean() - yp.mean()
        if abs(perm_stat) >= abs(obs):
            count += 1

    p_value = (count + 1) / (n_perm + 1)

    return DescriptiveResult(
        name="exact_perm_test",
        value=float(p_value),
        extra={
            "p_value": float(p_value),
            "observed_stat": obs,
            "n_perm": n_perm,
            "stat_type": stat,
            "n_x": nx,
            "n_y": len(y),
        },
    )


def cheatsheet() -> str:
    return "exact_perm_test({}) -> Exact permutation test for two-sample location."
