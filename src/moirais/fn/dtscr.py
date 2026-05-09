# moirais.fn — function file (hadesllm/moirais)
"""Scree test dimensionality.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtscr(data=None, n=50):
    """Scree test dimensionality.

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


short = "dtscr"
alias = "dtscr"
quote = "The spice must flow. -- Paul Atreides"
dtscr = dtscr


def cheatsheet() -> str:
    return "dtscr({}) -> Scree test dimensionality."
