# morie.fn — function file (hadesllm/morie)
"""DML: volatility effect on outcome."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def otis_dml_volatility(
    df,
    *,
    outcome_col: str = "recidivism",
    volatility_col: str = "volatility",
    covariate_cols: list[str] | None = None,
    n_folds: int = 2,
) -> ESRes:
    """DML for volatility treatment effect on outcome.

    Parameters
    ----------
    df : DataFrame
    outcome_col, volatility_col : str
    covariate_cols : list of str, optional
    n_folds : int

    Returns
    -------
    ESRes
    """
    y = np.asarray(df[outcome_col], dtype=float)
    d = np.asarray(df[volatility_col], dtype=float)
    if covariate_cols is None:
        covariate_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in (outcome_col, volatility_col)
        ]
    X = np.asarray(df[covariate_cols], dtype=float) if covariate_cols else np.ones((len(y), 1))
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    t_num, t_den = 0.0, 0.0
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        Xtr = np.column_stack([np.ones(len(tr)), X[tr]])
        Xte = np.column_stack([np.ones(len(te)), X[te]])
        by = np.linalg.lstsq(Xtr, y[tr], rcond=None)[0]
        bd = np.linalg.lstsq(Xtr, d[tr], rcond=None)[0]
        yr = y[te] - Xte @ by
        dr = d[te] - Xte @ bd
        t_num += dr @ yr
        t_den += dr @ dr
    theta = t_num / max(t_den, 1e-10)
    se = np.sqrt(1.0 / max(t_den, 1e-10))
    return ESRes(
        measure="otis_dml_volatility",
        estimate=float(theta),
        ci_lower=float(theta - 1.96 * se),
        ci_upper=float(theta + 1.96 * se),
        se=float(se),
        n=n,
    )


odml5 = otis_dml_volatility


def cheatsheet() -> str:
    return "otis_dml_volatility({}) -> DML: volatility effect on outcome."
