# morie.fn -- function file (hadesllm/morie)
"""
Kernel smoothing spatial

Category: KrigFilt
"""

import numpy as np


def ksmsp(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """Kernel smoothing spatial

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


short = "ksmsp"
alias = "ksmsp"
quote = "In the midst of chaos, there is also opportunity. -- Sun Tzu"
ksmsp = ksmsp


def cheatsheet() -> str:
    return "ksmsp({}) -> Kernel smoothing spatial"
