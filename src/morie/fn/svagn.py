"""1D agenda setting equilibrium"""

import numpy as np

from ._containers import DescriptiveResult


def agenda_1d(data, *, method="default"):
    """1D agenda setting equilibrium

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
        name="svagn",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


agen = agenda_1d


def cheatsheet() -> str:
    return "agenda_1d({}) -> 1D agenda setting equilibrium"
