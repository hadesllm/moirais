# morie.fn — function file (hadesllm/morie)
"""Lifting scheme DWT (in-place, memory efficient)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The road up and the road down are the same thing. — Heraclitus"


def lifting_dwt(
    x: np.ndarray,
    wavelet: str = "db4",
) -> DescriptiveResult:
    """Lifting scheme DWT: in-place wavelet transform.

    Implements the lazy wavelet (split) then predict+update steps.
    For Haar (db1): predict = even sample, update = average.

    Parameters
    ----------
    x : array-like
        1-D input signal (padded to even length if odd).
    wavelet : str
        Currently 'haar'/'db1' for exact lifting; others use
        convolution fallback via ``dwt_decompose``.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``coeffs`` [approx, detail], ``method``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) % 2 != 0:
        x = np.append(x, 0.0)
    if wavelet in ("haar", "db1"):
        even = x[0::2].copy()
        odd = x[1::2].copy()
        detail = odd - even
        approx = even + detail / 2.0
        method = "lifting"
    else:
        from .dwtfn import dwt_decompose

        res = dwt_decompose(x, wavelet=wavelet, level=1)
        coeffs = res.extra["coeffs"]
        approx = coeffs[0]
        detail = coeffs[1]
        method = "convolution_fallback"
    return DescriptiveResult(
        name="lifting_dwt",
        value=float(len(approx)),
        extra={"coeffs": [approx, detail], "method": method},
    )


liftw = lifting_dwt


def cheatsheet() -> str:
    return "lifting_dwt({}) -> Lifting scheme DWT (in-place, memory efficient)."
