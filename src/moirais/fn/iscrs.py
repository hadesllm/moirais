# moirais.fn — function file (hadesllm/moirais)
"""Cross-issue salience.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def iscrs(data=None, n=50):
    """Cross-issue salience.

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


short = "iscrs"
alias = "iscrs"
quote = "The spice must flow. -- Paul Atreides"
iscrs = iscrs


def cheatsheet() -> str:
    return "iscrs({}) -> Cross-issue salience."
