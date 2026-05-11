# morie.fn — function file (hadesllm/morie)
"""Multi-party Nash-Rubinstein.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpcnr(data=None, n=50):
    """Multi-party Nash-Rubinstein.

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


short = "mpcnr"
alias = "mpcnr"
quote = "The spice must flow. -- Paul Atreides"
mpcnr = mpcnr


def cheatsheet() -> str:
    return "mpcnr({}) -> Multi-party Nash-Rubinstein."
