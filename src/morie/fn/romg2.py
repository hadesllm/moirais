# morie.fn — function file (hadesllm/morie)
"""Omega per subscale."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import ESRes


def omega_subscale(
    data: pd.DataFrame | np.ndarray,
    subscale_items: list[str] | list[int] | None = None,
) -> ESRes:
    """Omega reliability for a single subscale.

    omega = (sum(lambda))^2 / ((sum(lambda))^2 + sum(uniqueness))

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response data (select subscale columns only).
    subscale_items : list, optional
        Column names or indices for the subscale.

    Returns
    -------
    ESRes
        measure="omega_subscale".

    References
    ----------
    McDonald, R. P. (1999). Test Theory: A Unified Treatment. Erlbaum.
    """
    if isinstance(data, pd.DataFrame):
        if subscale_items is not None:
            X = data[subscale_items].dropna().to_numpy(dtype=np.float64)
        else:
            X = data.dropna().to_numpy(dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    n, k = X.shape
    R = np.corrcoef(X, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(-evals)
    loads = np.abs(evecs[:, idx[0]] * np.sqrt(max(evals[idx[0]], 0)))
    uniqueness = 1 - loads**2

    sum_l = loads.sum()
    omega = sum_l**2 / (sum_l**2 + uniqueness.sum())

    return ESRes(
        measure="omega_subscale",
        estimate=float(omega),
        n=n,
        extra={"k": k, "loadings": loads.tolist()},
    )


omega_sub = omega_subscale


def cheatsheet() -> str:
    return "omega_subscale({}) -> Omega per subscale."
