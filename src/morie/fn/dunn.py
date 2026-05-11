# morie.fn — function file (hadesllm/morie)
"""Dunn's post-hoc test. 'There is another.'"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def dunn_test(*groups: np.ndarray, method: str = "bonferroni") -> DescriptiveResult:
    """Dunn's nonparametric post-hoc pairwise comparisons.

    After a significant Kruskal-Wallis test, Dunn's test compares
    all pairs of groups using rank sums and a z approximation.
    P-values are adjusted via *method* (bonferroni, holm, or none).

    :param groups: Two or more 1-D arrays of observations.
    :param method: P-value adjustment: 'bonferroni', 'holm', or 'none'.
    :return: DescriptiveResult with pairwise z-stats and adjusted p-values.
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups.")
    arrays = [np.asarray(g, dtype=float).ravel() for g in groups]
    k = len(arrays)
    all_data = np.concatenate(arrays)
    N = len(all_data)
    ranks = stats.rankdata(all_data)

    group_ranks = []
    idx = 0
    for a in arrays:
        ni = len(a)
        group_ranks.append(ranks[idx : idx + ni])
        idx += ni

    mean_ranks = [float(r.mean()) for r in group_ranks]
    ns = [len(a) for a in arrays]

    pairs = []
    z_stats = []
    raw_pvals = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = mean_ranks[i] - mean_ranks[j]
            se = np.sqrt((N * (N + 1) / 12.0) * (1.0 / ns[i] + 1.0 / ns[j]))
            z = diff / (se + 1e-12)
            p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            pairs.append((i, j))
            z_stats.append(float(z))
            raw_pvals.append(float(p))

    m = len(raw_pvals)
    if method == "bonferroni":
        adj = [min(p * m, 1.0) for p in raw_pvals]
    elif method == "holm":
        order = np.argsort(raw_pvals)
        adj = [0.0] * m
        for rank_i, idx in enumerate(order):
            adj[idx] = min(raw_pvals[idx] * (m - rank_i), 1.0)
    else:
        adj = raw_pvals

    return DescriptiveResult(
        name="dunn",
        value=float(np.min(adj)) if adj else 1.0,
        extra={
            "pairs": pairs,
            "z_statistics": z_stats,
            "raw_pvalues": raw_pvals,
            "adjusted_pvalues": adj,
            "method": method,
            "k": k,
            "N": N,
        },
    )


dunn = dunn_test


def cheatsheet() -> str:
    return "dunn_test({}) -> Dunn's post-hoc test. 'There is another.'"
