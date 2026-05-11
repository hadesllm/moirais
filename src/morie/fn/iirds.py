# morie.fn — function file (hadesllm/morie)
"""IIR filter design."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish."


def iir_design(order: int, cutoff, fs, ftype: str = "butter") -> DescriptiveResult:
    """Design an IIR filter and return b, a coefficients.

    Parameters
    ----------
    order : int
        Filter order.
    cutoff : float
        Cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    ftype : str
        Filter type: 'butter', 'cheby1', 'cheby2', 'ellip'. Default 'butter'.

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import iirfilter

    Wn = float(cutoff) / (fs / 2.0)
    b, a = iirfilter(order, Wn, btype="low", ftype=ftype, output="ba")
    return DescriptiveResult(
        name="iir_design",
        value=float(order),
        extra={"b": b, "a": a, "order": order, "cutoff": cutoff, "ftype": ftype},
    )


iirds = iir_design


def cheatsheet() -> str:
    return "iir_design({}) -> IIR filter design."
