# morie.fn — function file (hadesllm/morie)
"""FIR filter design via windowed sinc."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def fir_design(numtaps: int, cutoff, fs, window: str = "hamming") -> DescriptiveResult:
    r"""Design an FIR filter using the window method.

    .. math::

        h(n) = w(n) \\cdot \\frac{\\sin(\\omega_c (n - M))}{\\pi (n - M)}

    Parameters
    ----------
    numtaps : int
        Number of filter taps (length of FIR).
    cutoff : float or array-like
        Cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    window : str
        Window type. Default 'hamming'.

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import firwin

    coeffs = firwin(numtaps, cutoff, fs=fs, window=window)
    return DescriptiveResult(
        name="fir_design",
        value=float(numtaps),
        extra={"coefficients": coeffs, "numtaps": numtaps, "cutoff": cutoff, "window": window},
    )


firds = fir_design


def cheatsheet() -> str:
    return "fir_design({}) -> FIR filter design via windowed sinc."
