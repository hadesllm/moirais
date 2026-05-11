# morie.fn — function file (hadesllm/morie)
"""Legislative polarity detection"""

import numpy as np

from ._containers import DescriptiveResult


def leg_polarity(data, *, method="default"):
    """Legislative polarity detection

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
        name="nmpol",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


leg_ = leg_polarity


def cheatsheet() -> str:
    return "leg_polarity({}) -> Legislative polarity detection"
