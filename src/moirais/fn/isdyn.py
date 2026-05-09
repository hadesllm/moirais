# moirais.fn — function file (hadesllm/moirais)
"""Dynamic salience model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def isdyn(data=None, n=50):
    """Dynamic salience model.

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


short = "isdyn"
alias = "isdyn"
quote = "The spice must flow. -- Paul Atreides"
isdyn = isdyn


def cheatsheet() -> str:
    return "isdyn({}) -> Dynamic salience model."
