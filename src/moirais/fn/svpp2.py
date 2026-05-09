"""Party position 2D (Laver-Hunt)"""

import numpy as np

from ._containers import DescriptiveResult


def party_pos_2d(data, *, method="default"):
    """Party position 2D (Laver-Hunt)

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
        name="svpp2",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


part = party_pos_2d


def cheatsheet() -> str:
    return "party_pos_2d({}) -> Party position 2D (Laver-Hunt)"
