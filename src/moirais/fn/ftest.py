# moirais.fn — function file (hadesllm/moirais)
"""
F-test spatial distribution

Category: SpatialPat
"""

import numpy as np


def ftest(points=None, n=100, window=(0, 100, 0, 100)):
    """F-test spatial distribution

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


short = "ftest"
alias = "ftest"
quote = "The spice must flow. -- Paul Atreides"
ftest = ftest


def cheatsheet() -> str:
    return "ftest({}) -> F-test spatial distribution"
