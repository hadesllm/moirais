# morie.fn -- function file (hadesllm/morie)
"""Convergent validity: AVE > 0.5 for each subscale."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._mapq_const import SUBSCALES
from morie.fn.ave import ave


def subscale_convergent(
    data: pd.DataFrame,
    *,
    subscales: dict[str, list[str]] | None = None,
) -> dict:
    """Convergent validity check: AVE >= 0.5 for each subscale.

    AVE >= 0.5 means the factor explains more variance in its items
    than is due to measurement error (Fornell & Larcker, 1981).

    Parameters
    ----------
    data : DataFrame
        Item response data.
    subscales : dict, optional
        Subscale name -> list of item column names. Default: MAPQ.

    Returns
    -------
    dict
        Keys are subscale names, values are dicts with 'ave' and 'pass'.

    References
    ----------
    Fornell, C. & Larcker, D.F. (1981). Evaluating structural equation
        models with unobservable variables. JMR, 18(1), 39-50.
    """
    subs = subscales if subscales is not None else SUBSCALES
    result = {}

    for name, items in subs.items():
        cols = [c for c in items if c in data.columns]
        if len(cols) < 2:
            result[name] = {"ave": 0.0, "pass": False}
            continue

        X = data[cols].dropna().to_numpy(dtype=np.float64)
        R = np.corrcoef(X, rowvar=False)
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(-evals)
        loads = evecs[:, idx[0]] * np.sqrt(max(evals[idx[0]], 0))
        a = ave(loads)
        result[name] = {"ave": float(a), "pass": a >= 0.5}

    return result


def cheatsheet() -> str:
    return "subscale_convergent({}) -> Convergent validity: AVE > 0.5 for each subscale."
