# moirais.fn — function file (hadesllm/moirais)
"""
Source apportionment spatial

Category: EnvStat
"""

import numpy as np


def ensrc(data=None, coords=None, n=50):
    """Source apportionment spatial

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


short = "ensrc"
alias = "ensrc"
quote = "Equivalent exchange. -- Elric brothers"
ensrc = ensrc


def cheatsheet() -> str:
    return "ensrc({}) -> Source apportionment spatial"
