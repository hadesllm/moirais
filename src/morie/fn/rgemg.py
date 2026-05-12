# morie.fn -- function file (hadesllm/morie)
"""EMG RMS envelope -- Rangayyan Ch 8."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_emg_rms"]


def rangayyan_emg_rms(x, window=64, fs=1.0):
    """Sliding-window RMS envelope.

    RMS[n] = sqrt( (1/W) Σ_{k=n-W+1}^{n} x[k]² ).

    Parameters
    ----------
    x : array-like
    window : int
        Window length in samples.
    fs : float
        Sampling rate (Hz, only for reporting).

    Returns
    -------
    RichResult with keys ``rms``, ``window``, ``fs``, ``mean_rms``.

    References
    ----------
    Rangayyan Ch 8.
    """
    x = np.asarray(x, dtype=float).ravel()
    W = int(window)
    if W < 1:
        raise ValueError("window must be >= 1")
    sq = x ** 2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    rms = np.full_like(x, np.nan)
    for i in range(W - 1, x.size):
        rms[i] = np.sqrt((csum[i + 1] - csum[i + 1 - W]) / W)
    if x.size >= W:
        rms[:W - 1] = rms[W - 1]
    res = RichResult(
        title="EMG RMS envelope",
        summary_lines=[
            ("Window (samples)", W),
            ("Fs (Hz)", float(fs)),
            ("Mean RMS", float(np.nanmean(rms))),
            ("Max RMS", float(np.nanmax(rms))),
        ],
        interpretation=f"Sliding-window RMS, W={W} samples ({W / fs:.3g} s).",
        payload={"rms": rms, "window": W, "fs": float(fs),
                 "mean_rms": float(np.nanmean(rms))},
    )
    return with_describe_pointer(res, "rgemg")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_emg_rms(rng.standard_normal(500), window=32)
# >>> r["rms"].shape == (500,)
# True


def cheatsheet():
    return "rgemg: sliding-window RMS envelope -- Rangayyan Ch 8"
