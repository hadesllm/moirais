# morie.fn -- function file (hadesllm/morie)
"""Empirical Mode Decomposition into Intrinsic Mode Functions."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def emd_fn(x: np.ndarray, max_imfs: int = 10) -> DescriptiveResult:
    """Decompose signal into Intrinsic Mode Functions via EMD.

    :param x: 1-D input signal.
    :param max_imfs: Maximum number of IMFs to extract (default 10).
    :return: DescriptiveResult with number of IMFs as value and IMF list in extra.
    """
    from morie._decompose import emd

    x = np.asarray(x, dtype=float).ravel()
    imfs = emd(x, max_imfs=max_imfs)
    return DescriptiveResult(
        name="emd",
        value=len(imfs),
        extra={"imfs": imfs},
    )


emdfn = emd_fn


def cheatsheet() -> str:
    return "emd_fn({}) -> Empirical Mode Decomposition into Intrinsic Mode Functions."
