# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Cook's distance diagnostic. 'I shall endeavour to satisfy, sir.' -- Alfred Pennyworth"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def cooks_distance(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x_cols: list[str] | None = None,
    threshold: float | None = None,
) -> DescriptiveResult:
    """Cook's distance for each observation in an OLS regression.

    Identifies influential observations whose removal substantially changes
    the fitted regression.  Default threshold is 4/n (common rule of thumb).

    Parameters
    ----------
    data : DataFrame
        Input data with response and predictor columns.
    y : str
        Response column name.
    x_cols : list of str or None
        Predictor column names.  If None, uses all columns except *y*.
    threshold : float or None
        Threshold for flagging influential points.  Default 4/n.

    Returns
    -------
    DescriptiveResult
        ``value`` = number of influential points above threshold.
    """
    _validate_df(data, y)
    df = data.dropna()
    if x_cols is None:
        x_cols = [c for c in df.columns if c != y]
    if not x_cols:
        raise ValueError("No predictor columns found")
    _validate_df(df, *x_cols)
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x_cols].to_numpy(dtype=float)])
    n, p = X.shape
    if n <= p:
        raise ValueError(f"Need n > p; got n={n}, p={p}")
    hat_matrix = X @ np.linalg.solve(X.T @ X, X.T)
    h = np.diag(hat_matrix)
    y_hat = hat_matrix @ Y
    residuals = Y - y_hat
    mse = float(np.sum(residuals**2) / (n - p))
    cooks_d = (residuals**2 * h) / (p * mse * (1 - h) ** 2)
    if threshold is None:
        threshold = 4.0 / n
    influential = cooks_d > threshold
    n_influential = int(influential.sum())
    return DescriptiveResult(
        name="Cook's distance",
        value=n_influential,
        extra={
            "n": n,
            "p": p,
            "threshold": threshold,
            "n_influential": n_influential,
            "max_cooks_d": float(np.max(cooks_d)),
            "mean_cooks_d": float(np.mean(cooks_d)),
            "influential_indices": np.where(influential)[0].tolist()[:50],
            "cooks_d": cooks_d.tolist(),
        },
    )


alfrd = cooks_distance


def cheatsheet() -> str:
    return "cooks_distance({}) -> Cook's distance diagnostic. 'I shall endeavour to satisfy, s"
