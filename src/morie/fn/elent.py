# morie.fn -- function file (hadesllm/morie)
"""Electoral entry model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elent(data=None, n=50):
    """Electoral entry model.

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


short = "elent"
alias = "elent"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
elent = elent


def cheatsheet() -> str:
    return "elent({}) -> Electoral entry model."
