# moirais.fn — function file (hadesllm/moirais)
"""You have power over your mind — not outside events. — Marcus Aurelius"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import TestResult
from ._helpers import _validate_df


def permutation_two_sample(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    group: str = "group",
    n_perm: int = 9999,
    stat: str = "mean_diff",
    seed: int | None = None,
) -> TestResult:
    """Two-sample permutation test for equality of distributions.

    Computes the observed test statistic (difference in means by default) and
    generates a null distribution by randomly reassigning group labels.

    Parameters
    ----------
    data : DataFrame
        Input data.
    y : str
        Outcome column.
    group : str
        Binary group column (two unique values).
    n_perm : int
        Number of permutations.
    stat : str
        ``'mean_diff'`` or ``'median_diff'``.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    TestResult
    """
    _validate_df(data, y, group)
    df = data[[y, group]].dropna()
    groups = df[group].unique()
    if len(groups) != 2:
        raise ValueError(f"Need exactly 2 groups, got {len(groups)}")
    g0, g1 = groups[0], groups[1]
    y0 = df.loc[df[group] == g0, y].to_numpy(dtype=float)
    y1 = df.loc[df[group] == g1, y].to_numpy(dtype=float)
    func = np.mean if stat == "mean_diff" else np.median
    obs_stat = float(func(y1) - func(y0))
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([y0, y1])
    n1 = len(y1)
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null_stats[i] = func(perm[:n1]) - func(perm[n1:])
    p_value = float((np.abs(null_stats) >= np.abs(obs_stat)).sum() + 1) / (n_perm + 1)
    return TestResult(
        test_name="Permutation test (two-sample)",
        statistic=obs_stat,
        p_value=p_value,
        method=stat,
        n=len(df),
        extra={"n_perm": n_perm, "n0": len(y0), "n1": len(y1)},
    )


def cheatsheet() -> str:
    return "You have power over your mind — not outside events. — Marcus Aurelius"
