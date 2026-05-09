"""SNR dB to linear conversion."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "This is the way."


def snr_to_linear(snr_db, **kwargs) -> DescriptiveResult:
    """Convert SNR from decibels to linear scale.

    .. math::

        \\text{linear} = 10^{\\text{SNR}_{\\text{dB}} / 10}

    Parameters
    ----------
    snr_db : float
        Signal-to-noise ratio in decibels.

    Returns
    -------
    DescriptiveResult
    """
    snr_db = float(snr_db)
    linear = float(10.0 ** (snr_db / 10.0))
    return DescriptiveResult(
        name="snr_to_linear",
        value=linear,
        extra={"snr_db": snr_db, "linear": linear},
    )


snrdb = snr_to_linear


def cheatsheet() -> str:
    return "snr_to_linear({}) -> SNR dB to linear conversion."
