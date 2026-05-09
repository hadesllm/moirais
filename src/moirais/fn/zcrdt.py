"""Zero-crossing rate based onset detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Great, kid. Don't get cocky."


def zcr_detect(x, frame_len=256, hop=128, **kwargs) -> DescriptiveResult:
    """Zero-crossing rate (ZCR) based onset detection.

    Parameters
    ----------
    x : array-like
        Input signal.
    frame_len : int
        Frame length in samples. Default 256.
    hop : int
        Hop size in samples. Default 128.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    zcr_values = []
    positions = []

    for start in range(0, n - frame_len + 1, hop):
        frame = x[start : start + frame_len]
        crossings = np.sum(np.abs(np.diff(np.sign(frame))) > 0)
        zcr_values.append(float(crossings) / (frame_len - 1))
        positions.append(start + frame_len // 2)

    zcr_values = np.array(zcr_values)
    positions = np.array(positions, dtype=int)

    mean_zcr = float(np.mean(zcr_values)) if len(zcr_values) > 0 else 0.0

    return DescriptiveResult(
        name="zcr_detect",
        value=mean_zcr,
        extra={
            "zcr": zcr_values,
            "positions": positions,
            "n_frames": len(zcr_values),
            "frame_len": frame_len,
            "hop": hop,
        },
    )


zcrdt = zcr_detect


def cheatsheet() -> str:
    return "zcr_detect({}) -> Zero-crossing rate based onset detection."
