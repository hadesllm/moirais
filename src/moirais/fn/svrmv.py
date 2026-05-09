"""Rabinowitz-Macdonald intensity component"""

import numpy as np

from ._containers import DescriptiveResult


def rm_intensity(data, *, method="default"):
    """Rabinowitz-Macdonald intensity component

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
        name="svrmv",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


rm_i = rm_intensity


def cheatsheet() -> str:
    return "rm_intensity({}) -> Rabinowitz-Macdonald intensity component"
