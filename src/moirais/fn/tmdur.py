"""Signal time duration."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"


def time_duration(N: int, fs: float, **kwargs) -> DescriptiveResult:
    """Compute the time duration of a signal.

    .. math::

        T = \\frac{N}{f_s}

    Parameters
    ----------
    N : int
        Number of samples.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if N <= 0:
        raise ValueError("Number of samples must be positive.")
    T = N / fs
    return DescriptiveResult(
        name="time_duration",
        value=T,
        extra={"N": N, "fs": fs, "duration": T},
    )


tmdur = time_duration


def cheatsheet() -> str:
    return "time_duration({}) -> Signal time duration."
