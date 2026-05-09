# moirais.fn — function file (hadesllm/moirais)
"""CFA dimensionality test.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtcfa(data=None, n=50):
    """CFA dimensionality test.

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


short = "dtcfa"
alias = "dtcfa"
quote = "The spice must flow. -- Paul Atreides"
dtcfa = dtcfa


def cheatsheet() -> str:
    return "dtcfa({}) -> CFA dimensionality test."
