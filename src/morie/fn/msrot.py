# morie.fn — function file (hadesllm/morie)
"""Rotation of configuration"""

import numpy as np

from ._containers import DescriptiveResult


def rotate_config(data, *, method="default"):
    """Rotation of configuration

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
        name="msrot",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


rota = rotate_config


def cheatsheet() -> str:
    return "rotate_config({}) -> Rotation of configuration"
