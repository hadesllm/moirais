# moirais.fn — function file (hadesllm/moirais)
"""Stochastic median voter model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtst(data=None, n=50):
    """Stochastic median voter model.

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


short = "mvtst"
alias = "mvtst"
quote = "The spice must flow. -- Paul Atreides"
mvtst = mvtst


def cheatsheet() -> str:
    return "mvtst({}) -> Stochastic median voter model."
