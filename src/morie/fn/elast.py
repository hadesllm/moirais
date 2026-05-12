# morie.fn — function file (hadesllm/morie)
"""Price elasticity of demand. 'The ability to speak does not make you intelligent.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def elasticity(
    price: np.ndarray,
    quantity: np.ndarray,
) -> DescriptiveResult:
    r"""
    Compute arc price elasticity of demand.

    .. math::

        E_d = \\frac{\\Delta Q / \\bar{Q}}{\\Delta P / \\bar{P}}

    Uses the midpoint method for each consecutive pair.

    :param price: Array of price observations (at least 2).
    :param quantity: Array of corresponding quantity observations.
    :return: DescriptiveResult with mean elasticity and per-period values.
    :raises ValueError: If arrays differ in length or have < 2 elements.

    References
    ----------
    Varian, H. R. (2014). *Intermediate Microeconomics*. 9th ed.
    W. W. Norton.
    """
    p = np.asarray(price, dtype=np.float64)
    q = np.asarray(quantity, dtype=np.float64)
    if len(p) != len(q):
        raise ValueError("price and quantity must have equal length.")
    if len(p) < 2:
        raise ValueError("Need at least 2 observations.")

    dp = np.diff(p)
    dq = np.diff(q)
    p_mid = (p[:-1] + p[1:]) / 2.0
    q_mid = (q[:-1] + q[1:]) / 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        elas = (dq / q_mid) / (dp / p_mid)
    elas = np.where(np.isfinite(elas), elas, np.nan)

    mean_elas = float(np.nanmean(elas))

    return DescriptiveResult(
        name="Price Elasticity of Demand",
        value=mean_elas,
        extra={
            "elasticities": elas,
            "classification": "elastic" if abs(mean_elas) > 1 else ("unit" if abs(mean_elas) == 1 else "inelastic"),
            "n_periods": len(elas),
        },
    )


short = elasticity


def cheatsheet() -> str:
    return "elasticity({}) -> Price elasticity of demand. 'The ability to speak does not m"
