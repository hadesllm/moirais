# moirais.fn — function file (hadesllm/moirais)
"""
LOWESS spatial smoothing

Category: KrigFilt
"""

import numpy as np


def lowsp(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """LOWESS spatial smoothing

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


short = "lowsp"
alias = "lowsp"
quote = "Get in the robot, Shinji! -- Misato"
lowsp = lowsp


def cheatsheet() -> str:
    return "lowsp({}) -> LOWESS spatial smoothing"
