# moirais.fn — function file (hadesllm/moirais)
"""Issue agenda spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def isagd(data=None, n=50):
    """Issue agenda spatial.

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


short = "isagd"
alias = "isagd"
quote = "The spice must flow. -- Paul Atreides"
isagd = isagd


def cheatsheet() -> str:
    return "isagd({}) -> Issue agenda spatial."
