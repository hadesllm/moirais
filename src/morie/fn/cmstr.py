# morie.fn — function file (hadesllm/morie)
"""Strategic committee voting.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cmstr(data=None, n=50):
    """Strategic committee voting.

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


short = "cmstr"
alias = "cmstr"
quote = "The spice must flow. -- Paul Atreides"
cmstr = cmstr


def cheatsheet() -> str:
    return "cmstr({}) -> Strategic committee voting."
