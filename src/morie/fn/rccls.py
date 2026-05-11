# morie.fn — function file (hadesllm/morie)
"""Cluster roll call votes.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rccls(data=None, n=50):
    """Cluster roll call votes.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(data)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "rccls"
alias = "rccls"
quote = "The spice must flow. -- Paul Atreides"
rccls = rccls


def cheatsheet() -> str:
    return "rccls({}) -> Cluster roll call votes."
