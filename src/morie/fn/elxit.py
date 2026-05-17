# morie.fn -- function file (hadesllm/morie)
"""Electoral exit model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elxit(data=None, n=50):
    """Electoral exit model.

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


short = "elxit"
alias = "elxit"
quote = "The measure of a man is what he does with power. -- Plato"
elxit = elxit


def cheatsheet() -> str:
    return "elxit({}) -> Electoral exit model."
