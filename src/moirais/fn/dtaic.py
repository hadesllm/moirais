# moirais.fn — function file (hadesllm/moirais)
"""AIC dimensionality test.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtaic(data=None, n=50):
    """AIC dimensionality test.

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


short = "dtaic"
alias = "dtaic"
quote = "The spice must flow. -- Paul Atreides"
dtaic = dtaic


def cheatsheet() -> str:
    return "dtaic({}) -> AIC dimensionality test."
