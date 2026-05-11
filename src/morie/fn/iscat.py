# morie.fn — function file (hadesllm/morie)
"""Issue category salience.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def iscat(data=None, n=50):
    """Issue category salience.

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


short = "iscat"
alias = "iscat"
quote = "The spice must flow. -- Paul Atreides"
iscat = iscat


def cheatsheet() -> str:
    return "iscat({}) -> Issue category salience."
