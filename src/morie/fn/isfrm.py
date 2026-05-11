# morie.fn — function file (hadesllm/morie)
"""Issue framing model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def isfrm(data=None, n=50):
    """Issue framing model.

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


short = "isfrm"
alias = "isfrm"
quote = "The spice must flow. -- Paul Atreides"
isfrm = isfrm


def cheatsheet() -> str:
    return "isfrm({}) -> Issue framing model."
