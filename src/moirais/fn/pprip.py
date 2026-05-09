# moirais.fn — function file (hadesllm/moirais)
"""
Ripley correction PP

Category: PointProc
"""

import numpy as np


def pprip(points=None, n=80, window=(0, 1, 0, 1), intensity=None):
    """Ripley correction PP

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


short = "pprip"
alias = "pprip"
quote = "A Lannister always pays his debts. -- Tyrion"
pprip = pprip


def cheatsheet() -> str:
    return "pprip({}) -> Ripley correction PP"
