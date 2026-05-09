# moirais.fn — function file (hadesllm/moirais)
"""Three-party Hotelling competition.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hstrp(data=None, n=50):
    """Three-party Hotelling competition.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(np.asarray(data, dtype=float))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "hstrp"
alias = "hstrp"
quote = "The spice must flow. -- Paul Atreides"
hstrp = hstrp


def cheatsheet() -> str:
    return "hstrp({}) -> Three-party Hotelling competition."
