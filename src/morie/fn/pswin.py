# morie.fn -- function file (hadesllm/morie)
"""Win probability spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pswin(data=None, n=50):
    """Win probability spatial.

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


short = "pswin"
alias = "pswin"
quote = "No man ever steps in the same river twice. -- Heraclitus"
pswin = pswin


def cheatsheet() -> str:
    return "pswin({}) -> Win probability spatial."
