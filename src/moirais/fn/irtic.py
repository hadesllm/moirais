# moirais.fn — function file (hadesllm/moirais)
"""Item Characteristic Curves for IRT models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irtic(
    item_params: dict[str, dict],
    *,
    theta: np.ndarray | None = None,
    model: str = "2PL",
) -> pd.DataFrame:
    """Compute item characteristic curves P(theta) for each item.

    Returns the probability of a correct response at each theta value
    for every item in the model.

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
        Columns: "theta", plus one column per item with P(theta).

    References
    ----------
    Hambleton, R. K. & Swaminathan, H. (1985). Item Response Theory:
    Principles and Applications. Kluwer-Nijhoff.
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
        else:
            P = P_star

        result[name] = P

    return pd.DataFrame(result)


icc = irtic


def cheatsheet() -> str:
    return "irtic({}) -> Item Characteristic Curves for IRT models."
