# morie.fn -- function file (hadesllm/morie)
"""ER subscale reliability (alpha, omega, CR, AVE)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._mapq_const import SUBSCALES
from morie.fn.ave import ave
from morie.fn.crba import crba
from morie.fn.crel import crel
from morie.fn.mcdo import mcdo


def subscale_er(
    data: pd.DataFrame,
    *,
    items: list[str] | None = None,
) -> dict:
    """ER (Ethical Reservations) subscale reliability.

    Parameters
    ----------
    data : DataFrame
        Item response data.
    items : list of str, optional
        ER item column names. Default: ER1-ER5.

    Returns
    -------
    dict
        Keys: alpha, omega, cr, ave, n_items, n.
    """
    cols = items if items is not None else SUBSCALES["ER"]
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
    return "subscale_er({}) -> ER subscale reliability (alpha, omega, CR, AVE)."
