# morie.fn -- function file (hadesllm/morie)
"""
Ripley theta function

Category: PointProc
"""

import numpy as np


def pprth(points=None, n=80, window=(0, 1, 0, 1), intensity=None):
    """Ripley theta function

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


short = "pprth"
alias = "pprth"
quote = "The needs of the many outweigh the few. -- Spock"
pprth = pprth


def cheatsheet() -> str:
    return "pprth({}) -> Ripley theta function"
