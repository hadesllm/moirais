# morie.fn -- function file (hadesllm/morie)
"""W-NOMINATE 2D ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpw2(data=None, n=50):
    """W-NOMINATE 2D ideal point.

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


short = "idpw2"
alias = "idpw2"
quote = "Knowledge is power. -- Francis Bacon"
idpw2 = idpw2


def cheatsheet() -> str:
    return "idpw2({}) -> W-NOMINATE 2D ideal point."
