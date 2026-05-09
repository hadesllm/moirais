# moirais.fn — function file (hadesllm/moirais)
"""Factor analysis ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpfa(data=None, n=50):
    """Factor analysis ideal point.

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


short = "idpfa"
alias = "idpfa"
quote = "The spice must flow. -- Paul Atreides"
idpfa = idpfa


def cheatsheet() -> str:
    return "idpfa({}) -> Factor analysis ideal point."
