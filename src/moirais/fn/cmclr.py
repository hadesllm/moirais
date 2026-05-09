# moirais.fn — function file (hadesllm/moirais)
"""Closed rule committee.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cmclr(data=None, n=50):
    """Closed rule committee.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "cmclr"
alias = "cmclr"
quote = "The spice must flow. -- Paul Atreides"
cmclr = cmclr


def cheatsheet() -> str:
    return "cmclr({}) -> Closed rule committee."
