"""Smoothed pseudo Wigner-Ville distribution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def smoothed_pseudo_wvd_fn(
    x: np.ndarray,
    fs: float = 1.0,
    t_smooth: int = 11,
    f_smooth: int = 11,
) -> DescriptiveResult:
    """Compute smoothed pseudo Wigner-Ville distribution.

    :param x: 1-D input signal.
    :param fs: Sampling frequency in Hz (default 1.0).
    :param t_smooth: Time smoothing window length (default 11).
    :param f_smooth: Frequency smoothing window length (default 11).
    :return: DescriptiveResult with time-frequency distribution.
    """
    from morie._adaptive import smoothed_pseudo_wvd

    x = np.asarray(x, dtype=float).ravel()
    tfd, t, f = smoothed_pseudo_wvd(x, fs=fs, t_smooth=t_smooth, f_smooth=f_smooth)
    return DescriptiveResult(
        name="smoothed_pseudo_wvd",
        value=tfd.shape[0],
        extra={"tfd": tfd, "time": t, "frequencies": f},
    )


spwvd = smoothed_pseudo_wvd_fn


def cheatsheet() -> str:
    return "smoothed_pseudo_wvd_fn({}) -> Smoothed pseudo Wigner-Ville distribution."
