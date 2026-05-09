# moirais.fn — function file (hadesllm/moirais)
"""Issue salience weight.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def issal(data=None, n=50):
    """Issue salience weight.

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


short = "issal"
alias = "issal"
quote = "The spice must flow. -- Paul Atreides"
issal = issal


def cheatsheet() -> str:
    return "issal({}) -> Issue salience weight."
