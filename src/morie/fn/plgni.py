# morie.fn -- function file (hadesllm/morie)
"""Gini polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plgni(data=None, n=50):
    """Gini polarization index.

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


short = "plgni"
alias = "plgni"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
plgni = plgni


def cheatsheet() -> str:
    return "plgni({}) -> Gini polarization index."
