"""2D spatial polarization"""

import numpy as np

from ._containers import DescriptiveResult


def polarization_2d(data, *, method="default"):
    """2D spatial polarization

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
        name="svpl2",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


pola = polarization_2d


def cheatsheet() -> str:
    return "polarization_2d({}) -> 2D spatial polarization"
