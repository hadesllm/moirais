# morie.fn — function file (hadesllm/morie)
"""Nyquist frequency."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "In my experience, there is no such thing as luck."


def nyquist_freq(fs: float, **kwargs) -> DescriptiveResult:
    """Compute the Nyquist frequency.

    .. math::

        f_{\\text{Nyquist}} = \\frac{f_s}{2}

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    fnyq = fs / 2.0
    return DescriptiveResult(
        name="nyquist_freq",
        value=fnyq,
        extra={"fs": fs, "nyquist": fnyq},
    )


nqstf = nyquist_freq


def cheatsheet() -> str:
    return "nyquist_freq({}) -> Nyquist frequency."
