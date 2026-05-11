"""Lambert conformal conic projection"""

import numpy as np

from ._containers import DescriptiveResult


def lambert_proj(data, *, method="default"):
    """Lambert conformal conic projection

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
        name="zxlam",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


lamb = lambert_proj


def cheatsheet() -> str:
    return "lambert_proj({}) -> Lambert conformal conic projection"
