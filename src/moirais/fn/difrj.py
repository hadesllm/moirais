# moirais.fn — function file (hadesllm/moirais)
"""Raju's signed/unsigned area DIF measure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from moirais.fn._containers import DIFResult


def dif_raju_area(a_ref: np.ndarray | list, b_ref: np.ndarray | list, a_focal: np.ndarray | list, b_focal: np.ndarray | list, cdf=None, *, item_names: list[str] | None = None, alpha: float = 0.05) -> DIFResult:
    """Raju's signed and unsigned area measures for DIF.

    Parameters
    ----------
    a_ref, b_ref : array-like
        Discrimination and difficulty for reference group (k items).
    a_focal, b_focal : array-like
        Same for focal group.
    item_names : list[str], optional
        Item labels.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    DIFResult
        method="Raju", items with signed_area and unsigned_area.

    References
    ----------
    Raju, N. S. (1988). The area between two item characteristic curves.
    Psychometrika, 53(4), 495-502.
    """
    a_r = np.asarray(a_ref, dtype=np.float64)
    b_r = np.asarray(b_ref, dtype=np.float64)
    a_f = np.asarray(a_focal, dtype=np.float64)
    b_f = np.asarray(b_focal, dtype=np.float64)
    k = len(a_r)

    if item_names is None:
        item_names = [f"item_{j}" for j in range(k)]

    theta = np.linspace(-4, 4, 201)
    rows = []
    flagged = []
    for j in range(k):
        P_r = 1.0 / (1.0 + np.exp(-a_r[j] * (theta - b_r[j])))
        P_f = 1.0 / (1.0 + np.exp(-a_f[j] * (theta - b_f[j])))
        diff = P_r - P_f
        dt = theta[1] - theta[0]
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        signed = float(_trapz(diff, dx=dt))
        unsigned = float(_trapz(np.abs(diff), dx=dt))
        se = max(unsigned * 0.2, 0.01)
        z = abs(signed) / se
        p_val = 2 * (1 - sp.norm.cdf(z))
        rows.append(
            {
                "item": item_names[j],
                "signed_area": signed,
                "unsigned_area": unsigned,
                "z": float(z),
                "p_value": float(p_val),
            }
        )
        if p_val < alpha:
            flagged.append(item_names[j])

    return DIFResult(method="Raju", items=pd.DataFrame(rows), flagged=flagged)


raju_dif = dif_raju_area


def cheatsheet() -> str:
    return "dif_raju_area({}) -> Raju's signed/unsigned area DIF measure."
