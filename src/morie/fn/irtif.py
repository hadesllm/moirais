# morie.fn — function file (hadesllm/morie)
"""Item Information Function for IRT models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irtif(
    item_params: dict[str, dict],
    *,
    theta: np.ndarray | None = None,
    model: str = "2PL",
) -> pd.DataFrame:
    """Compute item information functions I(theta) for each item.

    For 1PL/2PL: I(theta) = a^2 * P(theta) * Q(theta)
    For 3PL:     I(theta) = a^2 * ((P - c)/(1 - c))^2 * Q / P

    Parameters
    ----------
    item_params : dict
        {item_name: {"a": float, "b": float, "c": float (optional)}}.
        Output from irt1p, irt2p, irt3p.
    theta : ndarray, optional
        Theta grid.  Default linspace(-4, 4, 81).
    model : str
        IRT model type: "1PL", "2PL", or "3PL" (default "2PL").

    Returns
    -------
    DataFrame
        Columns: "theta", plus one column per item with information values.

    References
    ----------
    Baker, F. B. & Kim, S.-H. (2004). Item Response Theory: Parameter
    Estimation Techniques (2nd ed.). Marcel Dekker.
    """
    if theta is None:
        theta = np.linspace(-4, 4, 81)
    theta = np.asarray(theta, dtype=np.float64).ravel()

    result = {"theta": theta}

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
            # Lord (1980) 3PL information formula
            P_adj = (P - c) / (1.0 - c)
            P_adj = np.clip(P_adj, 1e-10, 1.0)
            info = a**2 * P_adj**2 * Q / P
        else:
            # 1PL (a=1) or 2PL
            P = P_star
            Q = 1.0 - P
            info = a**2 * P * Q

        result[name] = info

    return pd.DataFrame(result)


item_info = irtif


def cheatsheet() -> str:
    return "irtif({}) -> Item Information Function for IRT models."
