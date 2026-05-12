# morie.fn -- function file (hadesllm/morie)
"""UA subscale average variance extracted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import ESRes


def subscale_ua_ave(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> ESRes:
    """Average Variance Extracted (AVE) for the UA subscale.

    Parameters
    ----------
    data : DataFrame or ndarray
    items : list[str], optional
        Default: UA1-UA5.

    Returns
    -------
    ESRes
        measure="AVE_UA".
    """
    if items is None:
        items = [f"UA{i}" for i in range(1, 6)]
    if isinstance(data, pd.DataFrame):
        X = data[items].dropna().to_numpy(dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    R = np.corrcoef(X, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(-evals)
    loads = np.abs(evecs[:, idx[0]] * np.sqrt(max(evals[idx[0]], 0)))

    return ESRes(
        measure="AVE_UA",
        estimate=float(np.mean(loads**2)),
        n=X.shape[0],
        extra={"loadings": loads.tolist(), "subscale": "UA"},
    )


ave_ua = subscale_ua_ave


def cheatsheet() -> str:
    return "subscale_ua_ave({}) -> UA subscale average variance extracted."
