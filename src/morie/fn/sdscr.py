# morie.fn — function file (hadesllm/morie)
"""Discriminant validity: sqrt(AVE) > inter-subscale r."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._mapq_const import SUBSCALES
from morie.fn.ave import ave


def subscale_discriminant(
    data: pd.DataFrame,
    *,
    subscales: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Fornell-Larcker discriminant validity test.

    For each pair of subscales, sqrt(AVE) of each should exceed
    their inter-subscale correlation (Fornell & Larcker, 1981).

    Parameters
    ----------
    data : DataFrame
        Item response data.
    subscales : dict, optional
        Subscale name -> list of item column names. Default: MAPQ.

    Returns
    -------
    DataFrame
        Columns: subscale_1, subscale_2, r, sqrt_ave_1, sqrt_ave_2, pass.

    References
    ----------
    Fornell, C. & Larcker, D.F. (1981). Evaluating structural equation
        models with unobservable variables. JMR, 18(1), 39-50.
    """
    subs = subscales if subscales is not None else SUBSCALES

    # Compute subscale scores
    scores = {}
    for name, items in subs.items():
        cols = [c for c in items if c in data.columns]
        if cols:
            scores[name] = data[cols].mean(axis=1).to_numpy()

    # Compute AVE for each subscale via single-factor loadings
    ave_vals = {}
    for name, items in subs.items():
        cols = [c for c in items if c in data.columns]
        if len(cols) < 2:
            ave_vals[name] = 0.0
            continue
        X = data[cols].dropna().to_numpy(dtype=np.float64)
        R = np.corrcoef(X, rowvar=False)
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(-evals)
        loads = evecs[:, idx[0]] * np.sqrt(max(evals[idx[0]], 0))
        ave_vals[name] = ave(loads)

    names = list(scores.keys())
    rows = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            r = float(np.corrcoef(scores[n1], scores[n2])[0, 1])
            sa1 = np.sqrt(ave_vals[n1])
            sa2 = np.sqrt(ave_vals[n2])
            passed = sa1 > abs(r) and sa2 > abs(r)
            rows.append(
                {
                    "subscale_1": n1,
                    "subscale_2": n2,
                    "r": r,
                    "sqrt_ave_1": float(sa1),
                    "sqrt_ave_2": float(sa2),
                    "pass": passed,
                }
            )

    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "subscale_discriminant({}) -> Discriminant validity: sqrt(AVE) > inter-subscale r."
