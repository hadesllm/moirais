# morie.fn — function file (hadesllm/morie)
"""EE subscale average variance extracted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import ESRes


def subscale_ee_ave(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> ESRes:
    """Average Variance Extracted (AVE) for the EE subscale.

    AVE = mean(lambda^2) where lambda are standardised factor loadings.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response data.
    items : list[str], optional
        Column names. Default: EE1-EE5.

    Returns
    -------
    ESRes
        measure="AVE_EE".
    """
    if items is None:
        items = [f"EE{i}" for i in range(1, 6)]
    if isinstance(data, pd.DataFrame):
        X = data[items].dropna().to_numpy(dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    R = np.corrcoef(X, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(-evals)
    loads = np.abs(evecs[:, idx[0]] * np.sqrt(max(evals[idx[0]], 0)))
    ave_val = float(np.mean(loads**2))

    return ESRes(
        measure="AVE_EE",
        estimate=ave_val,
        n=X.shape[0],
        extra={"loadings": loads.tolist(), "subscale": "EE"},
    )


ave_ee = subscale_ee_ave


def cheatsheet() -> str:
    return "subscale_ee_ave({}) -> EE subscale average variance extracted."
