"""Deegan-Packel power index"""

import numpy as np

from ._containers import DescriptiveResult


def deegan_packel(data, *, method="default"):
    """Deegan-Packel power index

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return DescriptiveResult(
        name="svdgn",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


deeg = deegan_packel


def cheatsheet() -> str:
    return "deegan_packel({}) -> Deegan-Packel power index"
