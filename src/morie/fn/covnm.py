# morie.fn -- function file (hadesllm/morie)
"""Covering number estimation (epsilon-net)."""

from __future__ import annotations

import numpy as np

__all__ = ["covnm"]


def covnm(
    points: np.ndarray,
    *,
    epsilons: np.ndarray | None = None,
    n_eps: int = 20,
    metric: str = "l2",
) -> dict:
    r"""
    Estimate covering numbers :math:`N(\varepsilon, \mathcal{T}, d)`.

    The covering number is the minimum number of :math:`\varepsilon`-balls
    needed to cover the set :math:`\mathcal{T}` under metric :math:`d`.

    Uses a greedy algorithm to construct an epsilon-net.

    :param points: Data matrix of shape (n, d).
    :param epsilons: Array of epsilon values. If None, auto-generated.
    :param n_eps: Number of epsilon values if auto-generated. Default 20.
    :param metric: ``"l2"`` (default) or ``"sup"``.
    :return: Dict with ``epsilons``, ``covering_numbers``, ``log_covering``.
    :raises ValueError: If points is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 6. Springer.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if points.size == 0:
        raise ValueError("points must be non-empty.")

    n = points.shape[0]

    if metric == "l2":
        def dist_fn(a, b):
            return np.sqrt(np.sum((a - b) ** 2, axis=-1))
    elif metric == "sup":
        def dist_fn(a, b):
            return np.max(np.abs(a - b), axis=-1)
    else:
        raise ValueError(f"metric must be 'l2' or 'sup', got '{metric}'.")

    if epsilons is None:
        all_dists = []
        for i in range(min(n, 100)):
            all_dists.append(np.max(dist_fn(points[i], points)))
        max_dist = max(all_dists) if all_dists else 1.0
        epsilons = np.linspace(max_dist / n_eps, max_dist, n_eps)

    covering_numbers = np.zeros(len(epsilons), dtype=int)
    for k, eps in enumerate(epsilons):
        covered = np.zeros(n, dtype=bool)
        count = 0
        for i in range(n):
            if not covered[i]:
                count += 1
                dists = dist_fn(points[i], points)
                covered[dists <= eps] = True
        covering_numbers[k] = count

    log_covering = np.log(np.maximum(covering_numbers.astype(float), 1.0))

    return {
        "epsilons": epsilons,
        "covering_numbers": covering_numbers,
        "log_covering": log_covering,
    }


def cheatsheet() -> str:
    return "covnm({points}) -> Covering number estimation (epsilon-net)."
