"""Turning points test for stationarity."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def turning_points_test(x, **kwargs) -> DescriptiveResult:
    """Apply the turning points test to signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    from moirais._filters import turning_points_test as _tp

    x = np.asarray(x, dtype=float)
    res = _tp(x)
    return DescriptiveResult(
        name="turning_points_test",
        value=res["z_statistic"],
        extra=res,
    )


trnpt = turning_points_test


def cheatsheet() -> str:
    return "turning_points_test({}) -> Turning points test for stationarity."
