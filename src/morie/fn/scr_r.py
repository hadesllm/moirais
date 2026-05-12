# morie.fn -- function file (hadesllm/morie)
"""ER subscale composite reliability (rho_c)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import ESRes


def subscale_er_composite_rel(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> ESRes:
    """Composite reliability (rho_c) for the ER subscale.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response data.
    items : list[str], optional
        Column names. Default: ER1-ER5.

    Returns
    -------
    ESRes
        measure="composite_reliability_ER".
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

    sum_l = loads.sum()
    sum_e = np.sum(1 - loads**2)
    cr = sum_l**2 / (sum_l**2 + sum_e)

    return ESRes(
        measure="composite_reliability_ER",
        estimate=float(cr),
        n=X.shape[0],
        extra={"loadings": loads.tolist(), "subscale": "ER"},
    )


cr_er = subscale_er_composite_rel


def cheatsheet() -> str:
    return "subscale_er_composite_rel({}) -> ER subscale composite reliability (rho_c)."
