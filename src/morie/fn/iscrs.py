# morie.fn -- function file (hadesllm/morie)
"""Cross-issue salience.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def iscrs(data=None, n=50):
    """Cross-issue salience.

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


short = "iscrs"
alias = "iscrs"
quote = "What is now proved was once only imagined. -- William Blake"
iscrs = iscrs


def cheatsheet() -> str:
    return "iscrs({}) -> Cross-issue salience."
