"""Signal padding."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "I know what I have to do but I don't know if I have the strength to do it."


def pad_signal(x, pad_len: int, mode: str = "zero") -> SignalResult:
    """Pad signal with zeros, mirror, or reflect.

    Parameters
    ----------
    x : array-like
        Input signal.
    pad_len : int
        Number of samples to pad on each side.
    mode : str
        'zero', 'mirror', or 'reflect'. Default 'zero'.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    pad_len = int(pad_len)
    if mode == "zero":
        y = np.pad(x, pad_len, mode="constant", constant_values=0.0)
    elif mode == "mirror":
        y = np.pad(x, pad_len, mode="symmetric")
    elif mode == "reflect":
        y = np.pad(x, pad_len, mode="reflect")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'zero', 'mirror', or 'reflect'.")
    return SignalResult(
        name="pad_signal",
        filtered=y,
        fs=0.0,
        n_samples=len(y),
        extra={"pad_len": pad_len, "mode": mode},
    )


sgpad = pad_signal


def cheatsheet() -> str:
    return "pad_signal({}) -> Signal padding."
