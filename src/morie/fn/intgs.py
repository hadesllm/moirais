# morie.fn — function file (hadesllm/morie)
"""Numerical integration of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "The belonging you seek is not behind you. It is ahead."


def integrate_signal(x, fs: float = 1.0) -> SignalResult:
    """Compute cumulative trapezoidal integration of signal *x*.

    .. math::

        y(n) = \\int_0^{n/f_s} x(t) \\, dt \\approx \\sum_{k=1}^{n}
               \\frac{x(k) + x(k-1)}{2 f_s}

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency (Hz). Default 1.0.

    Returns
    -------
    SignalResult
    """
    from scipy.integrate import cumulative_trapezoid

    x = np.asarray(x, dtype=float)
    dt = 1.0 / fs
    y = cumulative_trapezoid(x, dx=dt, initial=0.0)
    return SignalResult(
        name="integrate_signal",
        filtered=y,
        fs=fs,
        n_samples=len(y),
        extra={},
    )


intgs = integrate_signal


def cheatsheet() -> str:
    return "integrate_signal({}) -> Numerical integration of a signal."
