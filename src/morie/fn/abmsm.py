# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Agent-based model spatial

Category: SpatialPat
"""

import numpy as np


def abmsm(points=None, n=100, window=(0, 100, 0, 100)):
    """Agent-based model spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if points is None:
        points = np.column_stack(
            [
                np.random.default_rng(0).uniform(window[0], window[1], n),
                np.random.default_rng(1).uniform(window[2], window[3], n),
            ]
        )
    dists = np.sqrt(np.sum((points[:, None] - points[None, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    nn_dists = np.min(dists, axis=1)
    stat = float(np.mean(nn_dists))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(points), "mean_nn_dist": float(np.mean(nn_dists)), "window": window},
    )


short = "abmsm"
alias = "abmsm"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
abmsm = abmsm


def cheatsheet() -> str:
    return "abmsm({}) -> Agent-based model spatial"
