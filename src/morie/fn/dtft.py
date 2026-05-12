# morie.fn — function file (hadesllm/morie)
"""Discrete-Time Fourier Transform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def dtft_compute(x, omega=None, **kwargs) -> DescriptiveResult:
    r"""Compute the DTFT of signal *x* at specified frequencies.

    .. math::

        X(\\omega) = \\sum_{n=0}^{N-1} x(n) \\cdot e^{-j \\omega n}

    Parameters
    ----------
    x : array-like
        Input signal.
    omega : array-like or None
        Angular frequencies (rad/sample). Default: 512 points in [0, 2pi).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if omega is None:
        omega = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    omega = np.asarray(omega, dtype=float)
    n = np.arange(N)
    W = np.exp(-1j * np.outer(omega, n))
    X = W @ x
    return DescriptiveResult(
        name="dtft_compute",
        value=float(np.max(np.abs(X))),
        extra={"spectrum": X, "omega": omega, "N": N},
    )


dtft = dtft_compute


def cheatsheet() -> str:
    return "dtft_compute({}) -> Discrete-Time Fourier Transform."
