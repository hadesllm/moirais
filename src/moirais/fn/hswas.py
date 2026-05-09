# moirais.fn — function file (hadesllm/moirais)
"""Wasted-vote Hotelling.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hswas(data=None, n=50):
    """Wasted-vote Hotelling.

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


short = "hswas"
alias = "hswas"
quote = "The spice must flow. -- Paul Atreides"
hswas = hswas


def cheatsheet() -> str:
    return "hswas({}) -> Wasted-vote Hotelling."
