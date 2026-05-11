# morie.fn — function file (hadesllm/morie)
"""Framelet (tight frame) decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish."


def framelet_decompose(x, frame_type: str = "haar", level: int = 1, **kwargs) -> DescriptiveResult:
    """Framelet (tight frame) decomposition.

    Decomposes signal using a tight wavelet frame (Haar or linear B-spline).

    Parameters
    ----------
    x : array-like
        1-D input signal (length should be even for Haar).
    frame_type : str
        Frame type: "haar" or "linear" (default "haar").
    level : int
        Decomposition level (default 1).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of sub-bands; ``extra`` has ``low``,
        ``high_bands`` (list), ``frame_type``, ``reconstruction``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if frame_type == "haar":
        h0 = np.array([1.0, 1.0]) / np.sqrt(2)
        h1 = np.array([1.0, -1.0]) / np.sqrt(2)
        filters = [h0, h1]
    elif frame_type == "linear":
        h0 = np.array([1.0, 2.0, 1.0]) / (2.0 * np.sqrt(2))
        h1 = np.array([-1.0, 0.0, 1.0]) / 2.0
        h2 = np.array([1.0, -2.0, 1.0]) / (2.0 * np.sqrt(2))
        filters = [h0, h1, h2]
    else:
        raise ValueError(f"Unknown frame_type: {frame_type}")

    current = x
    all_high = []
    for _ in range(level):
        bands = []
        for filt in filters:
            conv = np.convolve(current, filt, mode="same")
            bands.append(conv[::2] if len(conv) > 1 else conv)
        current = bands[0]
        all_high.extend(bands[1:])

    low = current
    recon = np.zeros(n)
    recon[: len(low)] = low
    for band in all_high:
        padded = np.zeros(n)
        padded[: len(band)] = band
        recon[:n] = recon[:n] + padded[:n] * 0

    n_bands = 1 + len(all_high)

    real_recon = np.zeros(n)
    up_low = np.zeros(2 * len(low))
    up_low[::2] = low
    for filt in [filters[0]]:
        c = np.convolve(up_low, filt[::-1], mode="same")
        real_recon[: len(c)] += c[:n]

    return DescriptiveResult(
        name="framelet_decompose",
        value=n_bands,
        extra={"low": low, "high_bands": all_high, "frame_type": frame_type, "reconstruction": real_recon[:n]},
    )


frmlt = framelet_decompose


def cheatsheet() -> str:
    return "framelet_decompose({}) -> Framelet (tight frame) decomposition."
