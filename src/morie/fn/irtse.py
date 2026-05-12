# morie.fn -- function file (hadesllm/morie)
"""Standard error of theta at each ability level."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def irt_se_theta(
    item_params: dict,
    *,
    theta_grid: np.ndarray | None = None,
    model: str = "2PL",
) -> DescriptiveResult:
    """Compute SE(theta) at each point on the ability continuum.

    SE(theta) = 1 / sqrt(I(theta)) where I is test information.

    Parameters
    ----------
    item_params : dict
        {item_name: {"a": float, "b": float, ...}} per item.
    theta_grid : ndarray, optional
        Theta values. Default linspace(-4, 4, 81).
    model : str
        IRT model type: "1PL", "2PL", or "3PL".

    Returns
    -------
    DescriptiveResult
        value=dict with theta_grid and se arrays.
    """
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 81)

    info = np.zeros_like(theta_grid)
    for params in item_params.values():
        a = params.get("a", 1.0)
        b = params.get("b", 0.0)
        c = params.get("c", 0.0)
        logit = a * (theta_grid - b)
        P = c + (1 - c) / (1 + np.exp(-np.clip(logit, -700, 700)))
        P = np.clip(P, 1e-10, 1 - 1e-10)
        Q = 1 - P
        if model == "3PL" and c > 0:
            info += a**2 * ((P - c) / (1 - c)) ** 2 * Q / P
        else:
            info += a**2 * P * Q

    se = np.where(info > 1e-10, 1.0 / np.sqrt(info), np.nan)

    return DescriptiveResult(
        name="IRT SE(theta)",
        value={"theta": theta_grid.tolist(), "se": se.tolist()},
        extra={"model": model, "n_items": len(item_params)},
    )


se_theta = irt_se_theta


def cheatsheet() -> str:
    return "irt_se_theta({}) -> Standard error of theta at each ability level."
