# moirais.fn — function file (hadesllm/moirais)
"""Nash bargaining solution spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbnsh(data=None, n=50):
    """Nash bargaining solution spatial.

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


short = "sbnsh"
alias = "sbnsh"
quote = "The spice must flow. -- Paul Atreides"
sbnsh = sbnsh


def cheatsheet() -> str:
    return "sbnsh({}) -> Nash bargaining solution spatial."
