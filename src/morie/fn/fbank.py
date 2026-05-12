# morie.fn -- function file (hadesllm/morie)
"""QMF/CQF analysis+synthesis filter bank design."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "Rebellions are built on hope."


def filter_bank_design(wavelet: str = "db4") -> DescriptiveResult:
    """Design QMF analysis and synthesis filter bank for a given wavelet.

    Returns the four filters: analysis lowpass/highpass and
    synthesis lowpass/highpass satisfying perfect reconstruction.

    Parameters
    ----------
    wavelet : str
        Wavelet name (default 'db4').

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``lo_d``, ``hi_d`` (analysis),
        ``lo_r``, ``hi_r`` (synthesis), ``length``.
    """
    from .dwtfn import _wavelet_filter

    lo_d, hi_d = _wavelet_filter(wavelet)
    lo_r = lo_d[::-1]
    hi_r = hi_d[::-1]
    return DescriptiveResult(
        name="filter_bank_design",
        value=float(len(lo_d)),
        extra={
            "lo_d": lo_d,
            "hi_d": hi_d,
            "lo_r": lo_r,
            "hi_r": hi_r,
            "length": len(lo_d),
            "wavelet": wavelet,
        },
    )


fbank = filter_bank_design


def cheatsheet() -> str:
    return "filter_bank_design({}) -> QMF/CQF analysis+synthesis filter bank design."
