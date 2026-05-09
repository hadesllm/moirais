# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""
Microplastic spatial air

Category: AirBio
"""

import numpy as np


def abmcr(data=None, coords=None, n=50):
    """Microplastic spatial air

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


short = "abmcr"
alias = "abmcr"
quote = "Get in the robot, Shinji! -- Misato"
abmcr = abmcr


def cheatsheet() -> str:
    return "abmcr({}) -> Microplastic spatial air"
