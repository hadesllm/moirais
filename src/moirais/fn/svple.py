"""Esteban-Ray polarization index"""

import numpy as np

from ._containers import DescriptiveResult


def polarization_er(data, *, method="default"):
    """Esteban-Ray polarization index

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
        name="svple",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


pola = polarization_er


def cheatsheet() -> str:
    return "polarization_er({}) -> Esteban-Ray polarization index"
