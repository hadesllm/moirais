# morie.fn — function file (hadesllm/morie)
"""Federal median voter theorem.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtfe(data=None, n=50):
    """Federal median voter theorem.

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


short = "mvtfe"
alias = "mvtfe"
quote = "The spice must flow. -- Paul Atreides"
mvtfe = mvtfe


def cheatsheet() -> str:
    return "mvtfe({}) -> Federal median voter theorem."
