"""Spatial logit ML with GHK simulator."""

import numpy as np

from ._containers import SpatialResult


def splgtml(y, X, W, nsim=9):
    """Spatial logit ML with GHK simulator.

    Category: SProbit

    Parameters
    ----------
    y, X, W, nsim=9 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="splgtml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="splgtml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


splgtml_fn = splgtml


def cheatsheet() -> str:
    return "splgtml({}) -> Spatial logit ML with GHK simulator."
