# moirais.fn — function file (hadesllm/moirais)
"""Partisan salience model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ispas(data=None, n=50):
    """Partisan salience model.

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


short = "ispas"
alias = "ispas"
quote = "The spice must flow. -- Paul Atreides"
ispas = ispas


def cheatsheet() -> str:
    return "ispas({}) -> Partisan salience model."
