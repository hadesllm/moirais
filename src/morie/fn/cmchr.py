# morie.fn -- function file (hadesllm/morie)
"""Chair advantage spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cmchr(data=None, n=50):
    """Chair advantage spatial.

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


short = "cmchr"
alias = "cmchr"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
cmchr = cmchr


def cheatsheet() -> str:
    return "cmchr({}) -> Chair advantage spatial."
