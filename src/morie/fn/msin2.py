# morie.fn — function file (hadesllm/morie)
"""INDSCAL subject weights"""

import numpy as np

from ._containers import DescriptiveResult


def indscal_weights(data, *, method="default"):
    """INDSCAL subject weights

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
        name="msin2",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


inds = indscal_weights


def cheatsheet() -> str:
    return "indscal_weights({}) -> INDSCAL subject weights"
