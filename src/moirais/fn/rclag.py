# moirais.fn — function file (hadesllm/moirais)
"""Lagged roll call vote model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rclag(data=None, n=50):
    """Lagged roll call vote model.

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


short = "rclag"
alias = "rclag"
quote = "The spice must flow. -- Paul Atreides"
rclag = rclag


def cheatsheet() -> str:
    return "rclag({}) -> Lagged roll call vote model."
