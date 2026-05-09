"""
Bike lane spatial

Category: TransSp
"""

import numpy as np


def trbkl(flow_volume=None, travel_time=None, coords=None, n=50):
    """Bike lane spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if flow_volume is None:
        flow_volume = np.random.default_rng(0).poisson(500, n)
    if travel_time is None:
        travel_time = np.random.default_rng(1).uniform(5, 60, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    stat = float(np.mean(flow_volume))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n": len(flow_volume),
            "total_flow": int(np.sum(flow_volume)),
            "mean_travel_time": float(np.mean(travel_time)),
        },
    )


short = "trbkl"
alias = "trbkl"
quote = "The spice must flow. -- Paul Atreides"
trbkl = trbkl


def cheatsheet() -> str:
    return "trbkl({}) -> Bike lane spatial"
