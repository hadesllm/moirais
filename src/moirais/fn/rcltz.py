# moirais.fn — function file (hadesllm/moirais)
"""Luce-choice roll call model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcltz(data=None, n=50):
    """Luce-choice roll call model.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(data)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "rcltz"
alias = "rcltz"
quote = "The spice must flow. -- Paul Atreides"
rcltz = rcltz


def cheatsheet() -> str:
    return "rcltz({}) -> Luce-choice roll call model."
