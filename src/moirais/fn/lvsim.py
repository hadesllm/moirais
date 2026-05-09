# moirais.fn — function file (hadesllm/moirais)
"""
Levy flight simulation

Category: SpatialPat
"""

import numpy as np


def lvsim(points=None, n=100, window=(0, 100, 0, 100)):
    """Levy flight simulation

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


short = "lvsim"
alias = "lvsim"
quote = "A lesson without pain is meaningless. -- Edward"
lvsim = lvsim


def cheatsheet() -> str:
    return "lvsim({}) -> Levy flight simulation"
