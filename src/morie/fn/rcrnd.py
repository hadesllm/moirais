# morie.fn -- function file (hadesllm/morie)
"""Random utility roll call.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcrnd(data=None, n=50):
    """Random utility roll call.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(data)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "rcrnd"
alias = "rcrnd"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
rcrnd = rcrnd


def cheatsheet() -> str:
    return "rcrnd({}) -> Random utility roll call."
