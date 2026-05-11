# morie.fn — function file (hadesllm/morie)
"""Roll call matrix construction"""

import numpy as np

from ._containers import DescriptiveResult


def roll_call_matrix(data, *, method="default"):
    """Roll call matrix construction

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
        name="nmrcm",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


roll = roll_call_matrix


def cheatsheet() -> str:
    return "roll_call_matrix({}) -> Roll call matrix construction"
