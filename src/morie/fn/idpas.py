# morie.fn -- function file (hadesllm/morie)
"""Agenda-setter ideal point.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def idpas(data=None, n=50):
    """Agenda-setter ideal point.

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


short = "idpas"
alias = "idpas"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
idpas = idpas


def cheatsheet() -> str:
    return "idpas({}) -> Agenda-setter ideal point."
