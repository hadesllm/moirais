# morie.fn -- function file (hadesllm/morie)
"""Test Information Function for IRT models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irttf(
    item_params: dict[str, dict],
    *,
    theta: np.ndarray | None = None,
    model: str = "2PL",
) -> pd.DataFrame:
    """Compute the test information function and SE(theta).

    Test information is the sum of item information across all items.
    SE(theta) = 1 / sqrt(I(theta)).

    Parameters
    ----------
    item_params : dict
        {item_name: {"a": float, "b": float, "c": float (optional)}}.
    theta : ndarray, optional
        Theta grid.  Default linspace(-4, 4, 81).
    model : str
        IRT model type: "1PL", "2PL", or "3PL" (default "2PL").

    Returns
    -------
    DataFrame
        Columns: "theta", "information", "se_theta".

    References
    ----------
    Lord, F. M. (1980). Applications of Item Response Theory to Practical
    Testing Problems. Lawrence Erlbaum Associates.
    """
    if theta is None:
        theta = np.linspace(-4, 4, 81)
    theta = np.asarray(theta, dtype=np.float64).ravel()

    total_info = np.zeros_like(theta)

    for name, params in item_params.items():
        a = params.get("a", 1.0)
        b = params.get("b", 0.0)
        c = params.get("c", 0.0)

        logit = np.clip(a * (theta - b), -700, 700)
        P_star = 1.0 / (1.0 + np.exp(-logit))

        if model == "3PL" and c > 0:
            P = c + (1.0 - c) * P_star
            P = np.clip(P, 1e-10, 1.0 - 1e-10)
            Q = 1.0 - P
            P_adj = np.clip((P - c) / (1.0 - c), 1e-10, 1.0)
            info = a**2 * P_adj**2 * Q / P
        else:
            P = P_star
            Q = 1.0 - P
            info = a**2 * P * Q

        total_info += info

    se = np.where(total_info > 1e-10, 1.0 / np.sqrt(total_info), np.nan)

    return pd.DataFrame(
        {
            "theta": theta,
            "information": total_info,
            "se_theta": se,
        }
    )


test_info = irttf


def cheatsheet() -> str:
    return "irttf({}) -> Test Information Function for IRT models."
