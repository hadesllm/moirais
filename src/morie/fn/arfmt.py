# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""AR-based formant frequency extraction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ar_formant_extraction_fn(
    x: np.ndarray,
    fs: float = 8000.0,
    order: int = 12,
) -> DescriptiveResult:
    """Extract formant frequencies from speech via AR pole analysis.

    :param x: 1-D speech signal.
    :param fs: Sampling frequency in Hz (default 8000).
    :param order: AR model order (default 12).
    :return: DescriptiveResult with formant frequencies in extra.
    """
    from morie._biomodel import ar_formant_extraction

    x = np.asarray(x, dtype=float).ravel()
    formants = ar_formant_extraction(x, fs=fs, order=order)
    return DescriptiveResult(
        name="ar_formants",
        value=len(formants),
        extra={"formants": formants, "fs": fs, "order": order},
    )


arfmt = ar_formant_extraction_fn


def cheatsheet() -> str:
    return "ar_formant_extraction_fn({}) -> AR-based formant frequency extraction."
