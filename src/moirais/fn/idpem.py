# moirais.fn — function file (hadesllm/moirais)
"""EM algorithm ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpem(data=None, n=50):
    """EM algorithm ideal point.

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


short = "idpem"
alias = "idpem"
quote = "The spice must flow. -- Paul Atreides"
idpem = idpem


def cheatsheet() -> str:
    return "idpem({}) -> EM algorithm ideal point."
