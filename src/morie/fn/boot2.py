# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Two-sample bootstrap test."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def bootstrap_two_sample(
    x,
    y,
    *,
    statistic: str = "mean",
    n_boot: int = 9999,
    seed: int = 42,
    alternative: str = "two-sided",
) -> TestResult:
    """Two-sample bootstrap hypothesis test.

    Tests H0: no difference in the chosen statistic between
    the two groups via permutation-style resampling.

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    statistic : str
        ``"mean"`` or ``"median"``.
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        RNG seed.
    alternative : str
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 1 or len(y) < 1:
        raise ValueError("Both samples must be non-empty.")

    stat_fn = np.mean if statistic == "mean" else np.median
    obs_diff = stat_fn(x) - stat_fn(y)
    combined = np.concatenate([x, y])
    n1 = len(x)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_boot):
        perm = rng.permutation(combined)
        d = stat_fn(perm[:n1]) - stat_fn(perm[n1:])
        if alternative == "two-sided":
            if abs(d) >= abs(obs_diff):
                count += 1
        elif alternative == "greater":
            if d >= obs_diff:
                count += 1
        else:
            if d <= obs_diff:
                count += 1

    pval = (count + 1) / (n_boot + 1)

    return TestResult(
        test_name="Two-sample bootstrap test",
        statistic=float(obs_diff),
        p_value=float(pval),
        n=len(x) + len(y),
        method=f"statistic={statistic}, alternative={alternative}",
        extra={"n_boot": n_boot, "n1": len(x), "n2": len(y)},
    )


boot2 = bootstrap_two_sample


def cheatsheet() -> str:
    return "bootstrap_two_sample(x, y) -> Two-sample bootstrap test."
