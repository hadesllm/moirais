# morie.fn -- function file (hadesllm/morie)
"""Nominal ideal point estimation.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpnm(data=None, n=50):
    """Nominal ideal point estimation.

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


short = "idpnm"
alias = "idpnm"
quote = "The spice must flow. -- Paul Atreides"
idpnm = idpnm


def cheatsheet() -> str:
    return "idpnm({}) -> Nominal ideal point estimation."
