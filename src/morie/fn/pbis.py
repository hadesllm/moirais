# morie.fn — function file (hadesllm/morie)
"""Point-biserial correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def point_biserial(binary: np.ndarray, continuous: np.ndarray) -> ESRes:
    """Point-biserial correlation between a dichotomous and continuous variable.

    Parameters
    ----------
    binary : array-like (0/1)
    continuous : array-like

    Returns
    -------
    ESRes
    """
    binary = np.asarray(binary, dtype=float).ravel()
    continuous = np.asarray(continuous, dtype=float).ravel()
    mask = np.isfinite(binary) & np.isfinite(continuous)
    binary, continuous = binary[mask], continuous[mask]
    n = len(binary)
    if n < 4:
        raise ValueError("Need >= 4 observations.")

    r, p = sp_stats.pointbiserialr(binary, continuous)
    se = np.sqrt((1 - r**2) / (n - 2)) if n > 2 else 0.0

    return ESRes(
        measure="point_biserial_r",
        estimate=float(r),
        se=float(se),
        n=n,
        extra={"p_value": float(p)},
    )


pbis = point_biserial


def cheatsheet() -> str:
    return "point_biserial({}) -> Point-biserial correlation."
