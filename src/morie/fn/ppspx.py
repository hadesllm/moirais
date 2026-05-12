# morie.fn -- function file (hadesllm/morie)
"""
Space-time point process

Category: PointProc
"""

import numpy as np


def ppspx(points=None, n=80, window=(0, 1, 0, 1), intensity=None):
    """Space-time point process

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
    area = (window[1] - window[0]) * (window[3] - window[2])
    lam = len(points) / area if intensity is None else intensity
    stat = float(lam)
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n_points": len(points), "intensity": float(lam), "area": float(area), "window": window},
    )


short = "ppspx"
alias = "ppspx"
quote = "One does not simply walk. -- Boromir"
ppspx = ppspx


def cheatsheet() -> str:
    return "ppspx({}) -> Space-time point process"
