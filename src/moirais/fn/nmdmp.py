# moirais.fn — function file (hadesllm/moirais)
"""Party divergence measure"""

import numpy as np

from ._containers import DescriptiveResult


def party_diverge(data, *, method="default"):
    """Party divergence measure

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
        name="nmdmp",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


part = party_diverge


def cheatsheet() -> str:
    return "party_diverge({}) -> Party divergence measure"
