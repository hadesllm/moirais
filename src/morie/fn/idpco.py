# morie.fn -- function file (hadesllm/morie)
"""Cosponsorship ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpco(data=None, n=50):
    """Cosponsorship ideal point.

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


short = "idpco"
alias = "idpco"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
idpco = idpco


def cheatsheet() -> str:
    return "idpco({}) -> Cosponsorship ideal point."
