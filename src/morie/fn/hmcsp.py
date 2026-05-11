# morie.fn — function file (hadesllm/morie)
"""
Hamiltonian MC spatial

Category: SpatialPat
"""

import numpy as np


def hmcsp(points=None, n=100, window=(0, 100, 0, 100)):
    """Hamiltonian MC spatial

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


short = "hmcsp"
alias = "hmcsp"
quote = "Chaos is a ladder. -- Littlefinger"
hmcsp = hmcsp


def cheatsheet() -> str:
    return "hmcsp({}) -> Hamiltonian MC spatial"
