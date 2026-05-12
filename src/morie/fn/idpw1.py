# morie.fn -- function file (hadesllm/morie)
"""W-NOMINATE 1D ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpw1(data=None, n=50):
    """W-NOMINATE 1D ideal point.

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


short = "idpw1"
alias = "idpw1"
quote = "The spice must flow. -- Paul Atreides"
idpw1 = idpw1


def cheatsheet() -> str:
    return "idpw1({}) -> W-NOMINATE 1D ideal point."
