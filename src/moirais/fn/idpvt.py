# moirais.fn — function file (hadesllm/moirais)
"""Vote trading ideal point shift.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpvt(data=None, n=50):
    """Vote trading ideal point shift.

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


short = "idpvt"
alias = "idpvt"
quote = "The spice must flow. -- Paul Atreides"
idpvt = idpvt


def cheatsheet() -> str:
    return "idpvt({}) -> Vote trading ideal point shift."
