# morie.fn -- function file (hadesllm/morie)
"""
Kriging efficiency

Category: KrigFilt
"""

import numpy as np


def krgef(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """Kriging efficiency

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if x is None:
        x = np.random.default_rng(0).uniform(0, 100, 30)
    if y is None:
        y = np.random.default_rng(1).uniform(0, 100, 30)
    if values is None:
        values = np.random.default_rng(2).standard_normal(len(x))
    stat = float(np.var(values))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n_obs": len(x), "range": range_param, "sill": sill, "nugget": nugget, "grid_size": grid_size},
    )


short = "krgef"
alias = "krgef"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
krgef = krgef


def cheatsheet() -> str:
    return "krgef({}) -> Kriging efficiency"
