# moirais.fn — function file (hadesllm/moirais)
"""EE subscale composite reliability (rho_c)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import ESRes


def subscale_ee_composite_rel(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> ESRes:
    """Composite reliability (rho_c) for the EE subscale.

    CR = (sum(lambda))^2 / ((sum(lambda))^2 + sum(1 - lambda^2))

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response data.
    items : list[str], optional
        Column names. Default: EE1-EE5.

    Returns
    -------
    ESRes
        measure="composite_reliability_EE".
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

    sum_l = loads.sum()
    sum_e = np.sum(1 - loads**2)
    cr = sum_l**2 / (sum_l**2 + sum_e)

    return ESRes(
        measure="composite_reliability_EE",
        estimate=float(cr),
        n=X.shape[0],
        extra={"loadings": loads.tolist(), "subscale": "EE"},
    )


cr_ee = subscale_ee_composite_rel


def cheatsheet() -> str:
    return "subscale_ee_composite_rel({}) -> EE subscale composite reliability (rho_c)."
