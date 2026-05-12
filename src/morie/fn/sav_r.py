# morie.fn -- function file (hadesllm/morie)
"""ER subscale average variance extracted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import ESRes


def subscale_er_ave(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> ESRes:
    """Average Variance Extracted (AVE) for the ER subscale.

    Parameters
    ----------
    data : DataFrame or ndarray
    items : list[str], optional
        Default: ER1-ER5.

    Returns
    -------
    ESRes
        measure="AVE_ER".
    """
    if items is None:
        items = [f"ER{i}" for i in range(1, 6)]
    if isinstance(data, pd.DataFrame):
        X = data[items].dropna().to_numpy(dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    R = np.corrcoef(X, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(-evals)
    loads = np.abs(evecs[:, idx[0]] * np.sqrt(max(evals[idx[0]], 0)))

    return ESRes(
        measure="AVE_ER",
        estimate=float(np.mean(loads**2)),
        n=X.shape[0],
        extra={"loadings": loads.tolist(), "subscale": "ER"},
    )


ave_er = subscale_er_ave


def cheatsheet() -> str:
    return "subscale_er_ave({}) -> ER subscale average variance extracted."
