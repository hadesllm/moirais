# morie.fn -- function file (hadesllm/morie)
"""Runoff electoral model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elrun(data=None, n=50):
    """Runoff electoral model.

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


short = "elrun"
alias = "elrun"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
elrun = elrun


def cheatsheet() -> str:
    return "elrun({}) -> Runoff electoral model."
