# morie.fn — function file (hadesllm/morie)
"""Polyphase representation of filter."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The more you know, the more you realize you don't know. — Aristotle"


def polyphase_filter(
    h: np.ndarray,
    M: int = 2,
) -> DescriptiveResult:
    r"""Polyphase decomposition of a filter into M sub-filters.

    .. math::

        H(z) = \\sum_{k=0}^{M-1} z^{-k} E_k(z^M)

    Parameters
    ----------
    h : array-like
        Filter coefficients.
    M : int
        Number of polyphase components (default 2).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``components`` (list of M sub-filters), ``M``.
    """
    h = np.asarray(h, dtype=float).ravel()
    pad_len = int(np.ceil(len(h) / M)) * M - len(h)
    h_padded = np.append(h, np.zeros(pad_len))
    components = [h_padded[k::M] for k in range(M)]
    return DescriptiveResult(
        name="polyphase_filter",
        value=float(M),
        extra={"components": components, "M": M, "original_length": len(h)},
    )


prflt = polyphase_filter


def cheatsheet() -> str:
    return "polyphase_filter({}) -> Polyphase representation of filter."
