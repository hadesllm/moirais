# morie.fn -- function file (hadesllm/morie)
"""Roll call agreement score"""

import numpy as np

from ._containers import DescriptiveResult


def roll_call_agree(data, *, method="default"):
    """Roll call agreement score

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
        name="nmrca",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


roll = roll_call_agree


def cheatsheet() -> str:
    return "roll_call_agree({}) -> Roll call agreement score"
