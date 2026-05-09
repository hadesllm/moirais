# moirais.fn — function file (hadesllm/moirais)
"""
VOC spatial distribution

Category: EnvStat
"""

import numpy as np


def envoc(data=None, coords=None, n=50):
    """VOC spatial distribution

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


short = "envoc"
alias = "envoc"
quote = "It's over 9000! -- Vegeta"
envoc = envoc


def cheatsheet() -> str:
    return "envoc({}) -> VOC spatial distribution"
