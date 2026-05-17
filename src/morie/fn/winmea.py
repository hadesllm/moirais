# morie.fn -- function file (hadesllm/morie)
"""Winsorized mean: replace extreme values at both tails then average."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _extract_col


def winsorized_mean(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    proportion: float = 0.05,
) -> DescriptiveResult:
    """Winsorized mean: replace extreme values at both tails then average.

    Parameters
    ----------
    data : DataFrame or array
        Input data.
    col : str
        Column name if *data* is a DataFrame.
    proportion : float
        Fraction to winsorize on each tail (0 to 0.5 exclusive).

    Returns
    -------
    DescriptiveResult
        ``value`` = winsorized mean.
    """
    if not 0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    x = _extract_col(data, col)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 non-missing observations")
    k = int(np.floor(n * proportion))
    xs = np.sort(x)
    xs[:k] = xs[k]
    xs[n - k :] = xs[n - k - 1]
    wmean = float(np.mean(xs))
    wvar = float(np.var(xs, ddof=1))
    return DescriptiveResult(
        name="Winsorized mean",
        value=wmean,
        extra={
            "n": n,
            "k_trimmed": k,
            "proportion": proportion,
            "winsorized_variance": wvar,
            "raw_mean": float(np.mean(x)),
        },
    )


winmea = winsorized_mean


def cheatsheet() -> str:
    return 'winmea() -> Winsorized mean: replace extreme values at both tails then average'
