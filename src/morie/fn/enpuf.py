# morie.fn — function file (hadesllm/morie)
"""
Puff dispersion model

Category: EnvStat
"""

import numpy as np


def enpuf(data=None, coords=None, n=50):
    """Puff dispersion model

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


short = "enpuf"
alias = "enpuf"
quote = "El Psy Kongroo. -- Okabe"
enpuf = enpuf


def cheatsheet() -> str:
    return "enpuf({}) -> Puff dispersion model"
