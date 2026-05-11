# morie.fn — function file (hadesllm/morie)
"""Network degree distribution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def network_degree(adj: np.ndarray) -> DescriptiveResult:
    """Degree distribution of an unweighted/weighted network.

    Parameters
    ----------
    adj : (n, n) adjacency matrix (symmetric for undirected)

    Returns
    -------
    DescriptiveResult
    """
    adj = np.asarray(adj, dtype=float)
    n = adj.shape[0]
    if adj.shape != (n, n):
        raise ValueError("Adjacency matrix must be square.")

    adj_nodiag = adj.copy()
    np.fill_diagonal(adj_nodiag, 0)
    degrees = (adj_nodiag > 0).sum(axis=1).astype(int)
    strengths = adj.sum(axis=1)

    return DescriptiveResult(
        name="network_degree",
        value=float(np.mean(degrees)),
        extra={
            "degrees": degrees.tolist(),
            "mean_degree": float(np.mean(degrees)),
            "max_degree": int(np.max(degrees)),
            "min_degree": int(np.min(degrees)),
            "mean_strength": float(np.mean(strengths)),
            "density": float(np.sum(adj_nodiag > 0)) / (n * (n - 1)) if n > 1 else 0.0,
            "n": n,
        },
    )


netdg = network_degree


def cheatsheet() -> str:
    return "network_degree({}) -> Network degree distribution."
