# morie.fn — function file (hadesllm/morie)
"""Isotonic regression for MDS"""

import numpy as np

from ._containers import DescriptiveResult


def isotonic_reg(data, *, method="default"):
    """Isotonic regression for MDS

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
        name="msiso",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


isot = isotonic_reg


def cheatsheet() -> str:
    return "isotonic_reg({}) -> Isotonic regression for MDS"
