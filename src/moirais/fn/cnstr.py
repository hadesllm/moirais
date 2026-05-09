# moirais.fn — function file (hadesllm/moirais)
"""Anomaly removal via IQR. 'I'm a nasty piece of work.' -- John Constantine"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _extract_col


def iqr_exorcise(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    factor: float = 1.5,
) -> DescriptiveResult:
    """Remove anomalies (exorcise outliers) using the IQR fence method.

    Returns the cleaned data with outliers removed, along with diagnostics
    about what was removed.

    Parameters
    ----------
    data : DataFrame or array
        Input data.
    col : str
        Column name if *data* is a DataFrame.
    factor : float
        IQR multiplier for fence (1.5 = standard, 3.0 = extreme).

    Returns
    -------
    DescriptiveResult
        ``value`` = number of anomalies removed.
    """
    x = _extract_col(data, col)
    n_orig = len(x)
    if n_orig < 4:
        raise ValueError("Need at least 4 observations")
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (x >= lower) & (x <= upper)
    clean = x[mask]
    removed = x[~mask]
    return DescriptiveResult(
        name="IQR anomaly removal",
        value=int(len(removed)),
        extra={
            "n_original": n_orig,
            "n_kept": len(clean),
            "n_removed": len(removed),
            "pct_removed": round(len(removed) / n_orig * 100, 1),
            "lower_fence": lower,
            "upper_fence": upper,
            "iqr": iqr,
            "clean_mean": float(np.mean(clean)),
            "clean_sd": float(np.std(clean, ddof=1)),
            "removed_values": removed.tolist()[:20],
        },
    )


cnstr = iqr_exorcise


def cheatsheet() -> str:
    return "iqr_exorcise({}) -> Anomaly removal via IQR. 'I'm a nasty piece of work.' -- Joh"
