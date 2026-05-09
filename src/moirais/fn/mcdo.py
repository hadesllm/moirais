# moirais.fn — function file (hadesllm/moirais)
"""McDonald's omega total and hierarchical."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import OmgRes
from moirais.fn.crba import crba


def mcdo(
    data: pd.DataFrame | np.ndarray,
    nf: int = 1,
) -> OmgRes:
    """McDonald's omega total and hierarchical.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    nf : int
        Number of factors (default 1).

    Returns
    -------
    OmgRes
        Omega total, hierarchical, Cronbach alpha, and explained variance.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    R = np.corrcoef(X, rowvar=False)

    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    loads = evecs[:, :nf] * np.sqrt(np.maximum(evals[:nf], 0))
    comm = np.sum(loads**2, axis=1)
    uniq = 1 - comm
    omg_t = 1 - uniq.sum() / R.sum()
    omg_h = (loads[:, 0].sum() ** 2) / R.sum()

    a = crba(data)
    expvar = evals[:nf].sum() / evals.sum()

    return OmgRes(
        total=float(np.clip(omg_t, 0, 1)),
        hier=float(np.clip(omg_h, 0, 1)),
        alpha=a.raw,
        nf=nf,
        expvar=float(expvar),
    )


def cheatsheet() -> str:
    return "mcdo({}) -> McDonald's omega total and hierarchical."
