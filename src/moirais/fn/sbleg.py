# moirais.fn — function file (hadesllm/moirais)
"""Legislative bargaining spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbleg(data=None, n=50):
    """Legislative bargaining spatial.

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


short = "sbleg"
alias = "sbleg"
quote = "The spice must flow. -- Paul Atreides"
sbleg = sbleg


def cheatsheet() -> str:
    return "sbleg({}) -> Legislative bargaining spatial."
