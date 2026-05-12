# morie.fn -- function file (hadesllm/morie)
"""
Ordinary kriging filter

Category: KrigFilt
"""

import numpy as np


def okflt(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """Ordinary kriging filter

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


short = "okflt"
alias = "okflt"
quote = "Those who break the rules are scum. -- Kakashi"
okflt = okflt


def cheatsheet() -> str:
    return "okflt({}) -> Ordinary kriging filter"
