# moirais.fn — function file (hadesllm/moirais)
"""Coalition potential multi-party.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpclp(data=None, n=50):
    """Coalition potential multi-party.

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


short = "mpclp"
alias = "mpclp"
quote = "The spice must flow. -- Paul Atreides"
mpclp = mpclp


def cheatsheet() -> str:
    return "mpclp({}) -> Coalition potential multi-party."
