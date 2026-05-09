# moirais.fn — function file (hadesllm/moirais)
"""Abstention roll call model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcabs(data=None, n=50):
    """Abstention roll call model.

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


short = "rcabs"
alias = "rcabs"
quote = "The spice must flow. -- Paul Atreides"
rcabs = rcabs


def cheatsheet() -> str:
    return "rcabs({}) -> Abstention roll call model."
