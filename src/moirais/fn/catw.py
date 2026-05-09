# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Catenary curve fitting. 'Meow.' -- Catwoman"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def _catenary(x: np.ndarray, a: float, x0: float, y0: float) -> np.ndarray:
    return a * np.cosh((x - x0) / a) + y0 - a


def catenary_fit(
    data: pd.DataFrame,
    *,
    x: str = "x",
    y: str = "y",
) -> DescriptiveResult:
    """Fit a catenary curve y = a*cosh((x - x0)/a) + y0 - a to bivariate data.

    The catenary describes the shape of a flexible chain hanging under gravity.
    Useful in structural engineering, cable sag analysis, and curve morphology.

    Parameters
    ----------
    data : DataFrame
        Must contain *x* and *y* columns.
    x, y : str
        Column names.

    Returns
    -------
    DescriptiveResult
        ``value`` = fitted parameter *a* (catenary constant).
    """
    _validate_df(data, x, y)
    df = data[[x, y]].dropna()
    xv = df[x].to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float)
    if len(xv) < 4:
        raise ValueError("Need at least 4 data points for catenary fit")
    a0 = float(np.ptp(yv)) or 1.0
    x0_0 = float(np.mean(xv))
    y0_0 = float(np.min(yv))
    try:
        popt, pcov = curve_fit(_catenary, xv, yv, p0=[a0, x0_0, y0_0], maxfev=5000)
    except RuntimeError as exc:
        raise ValueError(f"Catenary fit did not converge: {exc}") from exc
    a_fit, x0_fit, y0_fit = popt
    y_pred = _catenary(xv, *popt)
    ss_res = float(np.sum((yv - y_pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="Catenary curve fit",
        value=float(a_fit),
        extra={
            "a": float(a_fit),
            "x0": float(x0_fit),
            "y0": float(y0_fit),
            "r_squared": r2,
            "rmse": float(np.sqrt(ss_res / len(xv))),
            "n": len(xv),
            "se": np.sqrt(np.diag(pcov)).tolist(),
        },
    )


catw = catenary_fit


def cheatsheet() -> str:
    return "_catenary({}) -> Catenary curve fitting. 'Meow.' -- Catwoman"
