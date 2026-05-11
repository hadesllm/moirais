# morie.fn — function file (hadesllm/morie)
"""Detrended fluctuation analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def detrended_fluctuation(
    x: np.ndarray,
    *,
    scales: np.ndarray | None = None,
) -> DescriptiveResult:
    """Detrended fluctuation analysis (DFA) alpha exponent.

    White noise: alpha ~ 0.5. Brownian motion: alpha ~ 1.5.

    :param x: 1-D input signal (length >= 16).
    :param scales: Array of window sizes. Auto-generated if None.
    :return: DescriptiveResult with alpha in ``value``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 16:
        return DescriptiveResult(name="dfa", value=float("nan"))

    y = np.cumsum(x - np.mean(x))

    if scales is None:
        min_s = 4
        max_s = n // 4
        n_scales = min(20, max_s - min_s + 1)
        scales = np.unique(
            np.logspace(
                np.log10(min_s),
                np.log10(max_s),
                n_scales,
            ).astype(int)
        )
        scales = scales[scales >= 4]

    fluct = np.zeros(len(scales))
    for i, s in enumerate(scales):
        n_seg = n // s
        if n_seg < 1:
            fluct[i] = np.nan
            continue
        rms_vals = []
        for j in range(n_seg):
            seg = y[j * s : (j + 1) * s]
            t = np.arange(s)
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            rms_vals.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluct[i] = np.mean(rms_vals)

    valid = (fluct > 0) & np.isfinite(fluct)
    if np.sum(valid) < 2:
        return DescriptiveResult(name="dfa", value=float("nan"))

    log_s = np.log(scales[valid].astype(float))
    log_f = np.log(fluct[valid])
    A = np.column_stack([log_s, np.ones(len(log_s))])
    alpha = float(np.linalg.lstsq(A, log_f, rcond=None)[0][0])

    return DescriptiveResult(
        name="dfa",
        value=alpha,
        extra={"scales": scales, "fluctuation": fluct},
    )


dfa = detrended_fluctuation


def cheatsheet() -> str:
    return "detrended_fluctuation({}) -> Detrended fluctuation analysis."
