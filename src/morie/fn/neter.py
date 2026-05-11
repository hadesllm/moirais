# morie.fn — function file (hadesllm/morie)
"""Erdos-Renyi random graph generator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def erdos_renyi(
    n: int,
    p: float,
    *,
    seed: int | None = None,
) -> DescriptiveResult:
    """Generate an Erdos-Renyi G(n, p) random graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge probability.
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1].")

    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p).astype(float)
    A = np.triu(A, 1)
    A = A + A.T

    n_edges = int(A.sum() / 2)
    degrees = A.sum(axis=1).astype(int)

    return DescriptiveResult(
        name="erdos_renyi",
        value=float(n_edges),
        extra={
            "n_nodes": n,
            "n_edges": n_edges,
            "p": p,
            "mean_degree": float(np.mean(degrees)),
            "density": float(2 * n_edges / (n * (n - 1))) if n > 1 else 0.0,
        },
    )


neter = erdos_renyi


def cheatsheet() -> str:
    return "erdos_renyi({}) -> Erdos-Renyi random graph generator."
