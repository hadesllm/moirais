# moirais.fn — function file (hadesllm/moirais)
"""Distance correlation."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def distance_correlation(x: np.ndarray, y: np.ndarray) -> ESRes:
    """Distance correlation (Szekely, Rizzo & Bakirov, 2007).

    Measures both linear and nonlinear dependence.

    Parameters
    ----------
    x, y : (n,) array-like

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        raise ValueError("Need >= 4 observations.")

    def _dcov2(a, b):
        A = np.abs(a[:, None] - a[None, :])
        B = np.abs(b[:, None] - b[None, :])
        A -= A.mean(axis=0, keepdims=True)
        A -= A.mean(axis=1, keepdims=True)
        A += A.mean()
        B -= B.mean(axis=0, keepdims=True)
        B -= B.mean(axis=1, keepdims=True)
        B += B.mean()
        return float(np.mean(A * B))

    dcov_xy = _dcov2(x, y)
    dcov_xx = _dcov2(x, x)
    dcov_yy = _dcov2(y, y)

    denom = np.sqrt(dcov_xx * dcov_yy)
    dcor = np.sqrt(max(dcov_xy, 0.0)) / np.sqrt(denom) if denom > 0 else 0.0

    return ESRes(
        measure="distance_correlation",
        estimate=float(dcor),
        n=n,
        extra={"dcov_xy": dcov_xy, "dcov_xx": dcov_xx, "dcov_yy": dcov_yy},
    )


dcorr = distance_correlation


def cheatsheet() -> str:
    return "distance_correlation({}) -> Distance correlation."
