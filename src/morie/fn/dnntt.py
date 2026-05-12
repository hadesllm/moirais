# morie.fn -- function file (hadesllm/morie)
"""Dunnett's test -- multiple treatment groups vs control."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def dunnett_test(
    control: np.ndarray,
    *treatment_groups: np.ndarray,
) -> DescriptiveResult:
    """Dunnett's test: each treatment vs control.

    Uses t-statistics with pooled variance (approximate, no
    multivariate-t critical values -- uses Bonferroni correction).

    Parameters
    ----------
    control : array-like
    *treatment_groups : array-like

    Returns
    -------
    DescriptiveResult
    """
    if len(treatment_groups) < 1:
        raise ValueError("Need >= 1 treatment group.")
    control = np.asarray(control, dtype=float)
    groups = [np.asarray(g, dtype=float) for g in treatment_groups]
    k = len(groups)

    n0 = len(control)
    ns = [len(g) for g in groups]
    N = n0 + sum(ns)
    grand_df = N - k - 1

    ss_pool = np.sum((control - control.mean()) ** 2)
    for g in groups:
        ss_pool += np.sum((g - g.mean()) ** 2)
    mse = ss_pool / grand_df if grand_df > 0 else 1.0

    results = []
    for i, g in enumerate(groups):
        diff = g.mean() - control.mean()
        se = np.sqrt(mse * (1 / len(g) + 1 / n0))
        t_stat = diff / se if se > 0 else 0.0
        p_raw = 2 * sp_stats.t.sf(abs(t_stat), grand_df)
        p_adj = min(p_raw * k, 1.0)
        results.append(
            {
                "group": i + 1,
                "diff": float(diff),
                "se": float(se),
                "t": float(t_stat),
                "p_adj": float(p_adj),
            }
        )

    return DescriptiveResult(
        name="dunnett",
        value=k,
        extra={"comparisons": results, "mse": float(mse), "df": grand_df, "n_control": n0, "correction": "bonferroni"},
    )


dnntt = dunnett_test


def cheatsheet() -> str:
    return "dunnett_test({}) -> Dunnett's test -- multiple treatment groups vs control."
