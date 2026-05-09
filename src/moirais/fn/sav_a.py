# moirais.fn — function file (hadesllm/moirais)
"""EA subscale average variance extracted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import ESRes


def subscale_ea_ave(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> ESRes:
    """Average Variance Extracted (AVE) for the EA subscale.

    Parameters
    ----------
    data : DataFrame or ndarray
    items : list[str], optional
        Default: EA1-EA5.

    Returns
    -------
    ESRes
        measure="AVE_EA".
    """
    if items is None:
        items = [f"EA{i}" for i in range(1, 6)]
    if isinstance(data, pd.DataFrame):
        X = data[items].dropna().to_numpy(dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    R = np.corrcoef(X, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(-evals)
    loads = np.abs(evecs[:, idx[0]] * np.sqrt(max(evals[idx[0]], 0)))

    return ESRes(
        measure="AVE_EA",
        estimate=float(np.mean(loads**2)),
        n=X.shape[0],
        extra={"loadings": loads.tolist(), "subscale": "EA"},
    )


ave_ea = subscale_ea_ave


def cheatsheet() -> str:
    return "subscale_ea_ave({}) -> EA subscale average variance extracted."
