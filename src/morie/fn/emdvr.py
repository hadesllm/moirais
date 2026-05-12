# morie.fn -- function file (hadesllm/morie)
"""Variance contribution of each IMF."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish."


def emd_variance_ratio(x, imfs) -> DescriptiveResult:
    """Compute variance contribution of each IMF.

    Parameters
    ----------
    x : array-like
        Original signal.
    imfs : list of array-like
        List of IMF arrays.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    total_var = float(np.var(x))
    ratios = []
    for imf in imfs:
        imf = np.asarray(imf, dtype=float).ravel()
        r = float(np.var(imf)) / (total_var + 1e-12)
        ratios.append(r)
    ratios = np.array(ratios)
    return DescriptiveResult(
        name="emd_variance_ratio",
        value=float(np.sum(ratios)),
        extra={"ratios": ratios, "total_variance": total_var, "n_imfs": len(imfs)},
    )


emdvr = emd_variance_ratio


def cheatsheet() -> str:
    return "emd_variance_ratio({}) -> Variance contribution of each IMF."
