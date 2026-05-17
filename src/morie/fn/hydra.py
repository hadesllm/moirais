# morie.fn -- function file (hadesllm/morie)
"""
Drainage network extraction

Category: HydroSp
"""

import numpy as np


def hydra(flow=None, precip=None, coords=None, n=50):
    """Drainage network extraction

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if flow is None:
        flow = np.abs(np.random.default_rng(0).standard_normal(n)) * 100
    if precip is None:
        precip = np.abs(np.random.default_rng(1).standard_normal(n)) * 50
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    stat = float(np.mean(flow))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(flow), "mean_flow": float(np.mean(flow)), "mean_precip": float(np.mean(precip))},
    )


short = "hydra"
alias = "hydra"
quote = "Statistics is the grammar of science. -- Karl Pearson"
hydra = hydra


def cheatsheet() -> str:
    return "hydra({}) -> Drainage network extraction"
