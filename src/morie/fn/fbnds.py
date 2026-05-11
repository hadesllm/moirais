# morie.fn — function file (hadesllm/morie)
"""Filter transition band bounds."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "The ability to speak does not make you intelligent."


def filter_bounds(order, cutoff, fs, **kwargs) -> DescriptiveResult:
    """Estimate filter transition band bounds.

    Uses the Kaiser approximation for transition bandwidth:
    delta_f ~ (A - 7.95) / (14.36 * order / fs) simplified to
    a rule-of-thumb: transition_bw ~ fs / (2 * order).

    Parameters
    ----------
    order : int
        Filter order.
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    if order <= 0:
        raise ValueError("Filter order must be positive.")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    transition_bw = fs / (2.0 * order)
    f_pass = cutoff - transition_bw / 2.0
    f_stop = cutoff + transition_bw / 2.0
    return DescriptiveResult(
        name="filter_bounds",
        value=transition_bw,
        extra={
            "order": order,
            "cutoff": cutoff,
            "fs": fs,
            "transition_bw": transition_bw,
            "f_pass": f_pass,
            "f_stop": f_stop,
        },
    )


fbnds = filter_bounds


def cheatsheet() -> str:
    return "filter_bounds({}) -> Filter transition band bounds."
