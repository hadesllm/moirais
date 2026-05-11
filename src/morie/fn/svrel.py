"""Random-effect logit spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svrel(data=None, n=50):
    """Random-effect logit spatial.

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


short = "svrel"
alias = "svrel"
quote = "The spice must flow. -- Paul Atreides"
svrel = svrel


def cheatsheet() -> str:
    return "svrel({}) -> Random-effect logit spatial."
