# moirais.fn — function file (hadesllm/moirais)
"""UA subscale reliability (alpha, omega, CR, AVE)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._mapq_const import SUBSCALES
from moirais.fn.ave import ave
from moirais.fn.crba import crba
from moirais.fn.crel import crel
from moirais.fn.mcdo import mcdo


def subscale_ua(
    data: pd.DataFrame,
    *,
    items: list[str] | None = None,
) -> dict:
    """UA (Utilitarian Attitudes) subscale reliability.

    Parameters
    ----------
    data : DataFrame
        Item response data.
    items : list of str, optional
        UA item column names. Default: UA1-UA5.

    Returns
    -------
    dict
        Keys: alpha, omega, cr, ave, n_items, n.
    """
    cols = items if items is not None else SUBSCALES["UA"]
    X = data[cols].dropna()
    arr = X.to_numpy(dtype=np.float64)

    a = crba(arr)
    o = mcdo(arr, nf=1)

    R = np.corrcoef(arr, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(-evals)
    loads = evecs[:, idx[0]] * np.sqrt(max(evals[idx[0]], 0))

    return {
        "alpha": a.raw,
        "omega": o.total,
        "cr": crel(loads),
        "ave": ave(loads),
        "n_items": len(cols),
        "n": len(X),
    }


def cheatsheet() -> str:
    return "subscale_ua({}) -> UA subscale reliability (alpha, omega, CR, AVE)."
