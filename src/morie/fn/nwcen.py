# morie.fn — function file (hadesllm/morie)
"""Node centrality measures for a network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def network_centrality(
    adjacency: np.ndarray,
    *,
    item_names: list[str] | None = None,
) -> DescriptiveResult:
    """Compute node centrality: strength, betweenness, closeness.

    Parameters
    ----------
    adjacency : ndarray
        Weighted adjacency matrix (p x p). Can be signed.
    item_names : list[str], optional

    Returns
    -------
    DescriptiveResult
        value=DataFrame with strength, betweenness, closeness per node.

    References
    ----------
    Opsahl, T., Agneessens, F., & Skvoretz, J. (2010). Node centrality
    in weighted networks. Social Networks, 32(3), 245-251.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    p = A.shape[0]

    if item_names is None:
        item_names = [f"node_{i}" for i in range(p)]

    strength = np.sum(np.abs(A), axis=1)

    dist = np.full((p, p), np.inf)
    np.fill_diagonal(dist, 0)
    abs_A = np.abs(A)
    for i in range(p):
        for j in range(p):
            if abs_A[i, j] > 1e-10:
                dist[i, j] = 1.0 / abs_A[i, j]

    for k in range(p):
        for i in range(p):
            for j in range(p):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    closeness = np.zeros(p)
    for i in range(p):
        reachable = dist[i, :] < np.inf
        reachable[i] = False
        n_reach = reachable.sum()
        if n_reach > 0:
            closeness[i] = n_reach / dist[i, reachable].sum()

    betweenness = np.zeros(p)
    for s in range(p):
        for t in range(s + 1, p):
            if dist[s, t] >= np.inf:
                continue
            for v in range(p):
                if v == s or v == t:
                    continue
                if abs(dist[s, v] + dist[v, t] - dist[s, t]) < 1e-10:
                    betweenness[v] += 1.0

    if p > 2:
        betweenness /= max((p - 1) * (p - 2) / 2, 1)

    df = pd.DataFrame(
        {
            "node": item_names,
            "strength": strength,
            "closeness": closeness,
            "betweenness": betweenness,
        }
    )

    return DescriptiveResult(
        name="Network centrality",
        value=df,
        extra={"n_nodes": p},
    )


centrality = network_centrality


def cheatsheet() -> str:
    return "network_centrality({}) -> Node centrality measures for a network."
