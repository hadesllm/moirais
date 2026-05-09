# moirais.fn — function file (hadesllm/moirais)
"""Kaiser-Meyer-Olkin sampling adequacy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import KmoRes


def kmo(data: pd.DataFrame | np.ndarray) -> KmoRes:
    """Kaiser-Meyer-Olkin measure of sampling adequacy.

    MSA > 0.6 adequate, > 0.8 meritorious (Kaiser, 1974).

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    KmoRes
        Overall MSA and per-item MSA values.
    """
    X = np.asarray(data, dtype=np.float64)
    R = np.corrcoef(X, rowvar=False)
    k = R.shape[0]

    try:
        Ri = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Ri = np.linalg.pinv(R)

    D = np.diag(1.0 / np.sqrt(np.diag(Ri)))
    Q = -D @ Ri @ D
    np.fill_diagonal(Q, 1.0)

    mask = ~np.eye(k, dtype=bool)
    sr2 = np.sum(R[mask] ** 2)
    sq2 = np.sum(Q[mask] ** 2)
    overall = sr2 / (sr2 + sq2)

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    items: dict[str, float] = {}
    for j in range(k):
        r2 = np.sum(R[j, mask[j]] ** 2)
        q2 = np.sum(Q[j, mask[j]] ** 2)
        items[names[j]] = float(r2 / (r2 + q2)) if (r2 + q2) > 0 else 0.0

    return KmoRes(msa=float(overall), items=items)


def cheatsheet() -> str:
    return "kmo({}) -> Kaiser-Meyer-Olkin sampling adequacy."
