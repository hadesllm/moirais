# morie.fn -- function file (hadesllm/morie)
"""OC for binary choice data with rating-to-pairwise conversion."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def oc_binary_choice(
    ratings,
    n_dims: int = 1,
    max_iter: int = 500,
    n_restarts: int = 10,
) -> DescriptiveResult:
    """Optimal Classification on binary choice data.

    Converts a rating matrix to pairwise binary choices before
    running OC, suitable for feeling thermometer or approval data.

    :param ratings: Rating matrix (respondents x stimuli).
    :param n_dims: Number of dimensions.
    :param max_iter: Maximum iterations per restart.
    :param n_restarts: Number of random restarts.
    :return: DescriptiveResult with PRE and ideal points in ``extra``.

    .. epigraph:: "Si vis pacem, para bellum." -- John Wick
    """
    from morie._spatial_voting import optimal_classification as _fn

    arr = np.asarray(ratings, dtype=float)
    n_resp, n_stim = arr.shape
    n_pairs = n_stim * (n_stim - 1) // 2
    votes = np.zeros((n_resp, n_pairs), dtype=int)
    col = 0
    for i in range(n_stim):
        for j in range(i + 1, n_stim):
            votes[:, col] = (arr[:, i] > arr[:, j]).astype(int)
            col += 1

    result = _fn(votes, n_dims=n_dims, max_iter=max_iter, n_restarts=n_restarts)
    result["n_stimuli"] = n_stim
    result["n_pairs"] = n_pairs
    return DescriptiveResult(
        name="oc_binary_choice",
        value=result["PRE"],
        extra=result,
    )


ocbin = oc_binary_choice


def cheatsheet() -> str:
    return "oc_binary_choice({}) -> OC for binary choice data with rating-to-pairwise conversion"
