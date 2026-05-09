# moirais.fn — function file (hadesllm/moirais)
"""Entry deterrence Hotelling.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsent(data=None, n=50):
    """Entry deterrence Hotelling.

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


short = "hsent"
alias = "hsent"
quote = "The spice must flow. -- Paul Atreides"
hsent = hsent


def cheatsheet() -> str:
    return "hsent({}) -> Entry deterrence Hotelling."
