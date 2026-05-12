"""Quantization SNR."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"


def snr_quantization(bits: int, **kwargs) -> DescriptiveResult:
    r"""Compute the theoretical quantization SNR.

    .. math::

        \\text{SNR}_q = 6.02 b + 1.76 \\;\\text{dB}

    Parameters
    ----------
    bits : int
        Number of quantization bits.

    Returns
    -------
    DescriptiveResult
    """
    if bits <= 0:
        raise ValueError("Number of bits must be positive.")
    snr_db = 6.02 * bits + 1.76
    return DescriptiveResult(
        name="snr_quantization",
        value=snr_db,
        extra={"bits": bits, "snr_db": snr_db},
    )


snrqt = snr_quantization


def cheatsheet() -> str:
    return "snr_quantization({}) -> Quantization SNR."
