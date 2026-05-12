# morie.fn -- function file (hadesllm/morie)
"""Frequency resolution."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "Study the past if you would define the future. -- Confucius"


def freq_resolution(fs: float, N: int, **kwargs) -> DescriptiveResult:
    r"""Compute the frequency resolution of a DFT.

    .. math::

        \\Delta f = \\frac{f_s}{N}

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    N : int
        Number of samples.

    Returns
    -------
    DescriptiveResult
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if N <= 0:
        raise ValueError("Number of samples must be positive.")
    df = fs / N
    return DescriptiveResult(
        name="freq_resolution",
        value=df,
        extra={"fs": fs, "N": N, "df": df},
    )


rsltn = freq_resolution


def cheatsheet() -> str:
    return "freq_resolution({}) -> Frequency resolution."
