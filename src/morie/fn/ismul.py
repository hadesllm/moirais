# morie.fn -- function file (hadesllm/morie)
"""Multi-issue salience.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ismul(data=None, n=50):
    """Multi-issue salience.

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


short = "ismul"
alias = "ismul"
quote = "Knowledge is power. -- Francis Bacon"
ismul = ismul


def cheatsheet() -> str:
    return "ismul({}) -> Multi-issue salience."
