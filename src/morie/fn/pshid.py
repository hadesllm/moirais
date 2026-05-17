# morie.fn -- function file (hadesllm/morie)
"""Hidden state spatial model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pshid(data=None, n=50):
    """Hidden state spatial model.

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


short = "pshid"
alias = "pshid"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
pshid = pshid


def cheatsheet() -> str:
    return "pshid({}) -> Hidden state spatial model."
