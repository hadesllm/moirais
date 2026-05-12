# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bracketing entropy of a function class."""

from __future__ import annotations

import numpy as np

__all__ = ["brent"]


def brent(
    data: np.ndarray,
    *,
    epsilons: np.ndarray | None = None,
    n_eps: int = 20,
    metric: str = "l2",
) -> dict:
    r"""
    Estimate bracketing entropy for a function class evaluated on data.

    The bracketing number :math:`N_{[]}(\varepsilon, \mathcal{F}, L_2(P))`
    is the minimum number of :math:`\varepsilon`-brackets needed to cover
    :math:`\mathcal{F}`. The bracketing entropy is its logarithm.

    This implementation estimates bracketing numbers by counting how many
    :math:`\varepsilon`-balls are needed to cover the observed function
    evaluations in :math:`L_2` norm.

    :param data: Matrix of shape (n_functions, n_points) -- each row is a
        function evaluated at n_points.
    :param epsilons: Array of epsilon values. If None, auto-generated.
    :param n_eps: Number of epsilon values if auto-generated. Default 20.
    :param metric: Distance metric, ``"l2"`` (default) or ``"sup"``.
    :return: Dict with ``epsilons``, ``bracketing_numbers``, ``log_bracketing``,
        ``entropy_integral``.
    :raises ValueError: If data is empty or metric unknown.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 6. Springer.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        raise ValueError("data must be non-empty.")
    if metric not in ("l2", "sup"):
        raise ValueError(f"metric must be 'l2' or 'sup', got '{metric}'.")

    n_funcs, n_pts = data.shape

    if metric == "l2":
        def dist(a, b):
            return np.sqrt(np.mean((a - b) ** 2))
    else:
        def dist(a, b):
            return np.max(np.abs(a - b))

    dists = np.zeros((n_funcs, n_funcs))
    for i in range(n_funcs):
        for j in range(i + 1, n_funcs):
            d = dist(data[i], data[j])
            dists[i, j] = d
            dists[j, i] = d

    if epsilons is None:
        max_dist = np.max(dists) if n_funcs > 1 else 1.0
        epsilons = np.linspace(max_dist / n_eps, max_dist, n_eps)

    bracketing_numbers = np.zeros(len(epsilons), dtype=int)
    for k, eps in enumerate(epsilons):
        covered = np.zeros(n_funcs, dtype=bool)
        count = 0
        for i in range(n_funcs):
            if not covered[i]:
                count += 1
                covered[dists[i] <= eps] = True
        bracketing_numbers[k] = max(count, 1)

    log_bracketing = np.log(bracketing_numbers.astype(float))

    integrand = np.sqrt(log_bracketing)
    entropy_integral = float(np.trapezoid(integrand, epsilons))

    return {
        "epsilons": epsilons,
        "bracketing_numbers": bracketing_numbers,
        "log_bracketing": log_bracketing,
        "entropy_integral": entropy_integral,
    }


def cheatsheet() -> str:
    return "brent({data}) -> Bracketing entropy of function class."
