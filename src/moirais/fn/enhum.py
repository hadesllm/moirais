# moirais.fn — function file (hadesllm/moirais)
"""
Humidity spatial mapping

Category: EnvStat
"""

import numpy as np


def enhum(data=None, coords=None, n=50):
    """Humidity spatial mapping

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "enhum"
alias = "enhum"
quote = "The world is cruel but beautiful. -- Mikasa"
enhum = enhum


def cheatsheet() -> str:
    return "enhum({}) -> Humidity spatial mapping"
