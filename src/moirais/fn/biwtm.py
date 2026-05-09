# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Biweight midcorrelation. 'Truly wonderful the mind of a child is.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def biweight_midcorrelation(x: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """Biweight midcorrelation (bicor) — robust alternative to Pearson r.

    Uses Tukey's biweight function to downweight outliers.  The
    measure has a 50% breakdown point while retaining high
    efficiency at the Gaussian model.  Widely used in
    genomics (WGCNA) and signal processing.

    :param x: 1-D numeric array.
    :param y: 1-D numeric array (same length as x).
    :return: DescriptiveResult with bicor estimate.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length.")

    med_x = np.median(x)
    med_y = np.median(y)
    mad_x = np.median(np.abs(x - med_x)) + 1e-12
    mad_y = np.median(np.abs(y - med_y)) + 1e-12

    u = (x - med_x) / (9.0 * mad_x)
    v = (y - med_y) / (9.0 * mad_y)

    wx = (1.0 - u**2) ** 2 * (np.abs(u) < 1.0)
    wy = (1.0 - v**2) ** 2 * (np.abs(v) < 1.0)

    xw = (x - med_x) * wx
    yw = (y - med_y) * wy

    num = np.sum(xw * yw)
    denom = np.sqrt(np.sum(xw**2) * np.sum(yw**2)) + 1e-12
    bicor = float(num / denom)

    return DescriptiveResult(
        name="biwtm",
        value=bicor,
        extra={"n": n, "mad_x": float(mad_x), "mad_y": float(mad_y)},
    )


biwtm = biweight_midcorrelation


def cheatsheet() -> str:
    return "biweight_midcorrelation({}) -> Biweight midcorrelation. 'Truly wonderful the mind of a chil"
