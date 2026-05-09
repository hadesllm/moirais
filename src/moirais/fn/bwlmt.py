# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Nyquist bandwidth limit analysis."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "There are no facts, only interpretations. — Friedrich Nietzsche"


def bandwidth_limit(fs, signal_bw, **kwargs) -> DescriptiveResult:
    """Analyse whether a signal bandwidth respects the Nyquist limit.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    signal_bw : float
        Signal bandwidth in Hz.

    Returns
    -------
    DescriptiveResult
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if signal_bw < 0:
        raise ValueError("Signal bandwidth must be non-negative.")
    nyquist = fs / 2.0
    margin = nyquist - signal_bw
    aliased = bool(signal_bw > nyquist)
    return DescriptiveResult(
        name="bandwidth_limit",
        value=nyquist,
        extra={
            "fs": fs,
            "signal_bw": signal_bw,
            "nyquist": nyquist,
            "margin": margin,
            "aliased": aliased,
        },
    )


bwlmt = bandwidth_limit


def cheatsheet() -> str:
    return "bandwidth_limit({}) -> Nyquist bandwidth limit analysis."
