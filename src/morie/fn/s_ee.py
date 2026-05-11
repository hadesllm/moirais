# morie.fn — function file (hadesllm/morie)
"""EE subscale reliability (alpha, omega, CR, AVE)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._mapq_const import SUBSCALES
from morie.fn.ave import ave
from morie.fn.crba import crba
from morie.fn.crel import crel
from morie.fn.mcdo import mcdo


def subscale_ee(
    data: pd.DataFrame,
    *,
    items: list[str] | None = None,
) -> dict:
    """EE (Experiential Engagement) subscale reliability.

    Parameters
    ----------
    data : DataFrame
        Item response data.
    items : list of str, optional
        EE item column names. Default: EE1-EE5.

    Returns
    -------
    dict
        Keys: alpha, omega, cr, ave, n_items, n.
    """
    cols = items if items is not None else SUBSCALES["EE"]
    X = data[cols].dropna()
    arr = X.to_numpy(dtype=np.float64)

    a = crba(arr)
    o = mcdo(arr, nf=1)

    # Factor loadings from single-factor EFA for CR/AVE
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
    return "subscale_ee({}) -> EE subscale reliability (alpha, omega, CR, AVE)."
