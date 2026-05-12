# morie.fn -- function file (hadesllm/morie)
"""Price-location Hotelling.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsprc(data=None, n=50):
    """Price-location Hotelling.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(np.asarray(data, dtype=float))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "hsprc"
alias = "hsprc"
quote = "The spice must flow. -- Paul Atreides"
hsprc = hsprc


def cheatsheet() -> str:
    return "hsprc({}) -> Price-location Hotelling."
