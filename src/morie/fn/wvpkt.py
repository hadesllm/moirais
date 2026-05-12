"""Wavelet packet decomposition (full tree)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The more you know, the more you realize you don't know. -- Aristotle"


def wavelet_packets(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Wavelet packet decomposition -- full binary tree.

    Unlike the standard DWT which only decomposes approximation coefficients,
    wavelet packets decompose both approximation and detail at every level.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name (default 'db4').
    level : int
        Decomposition level (default 3).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``tree`` (dict mapping (level, node) to coefficients),
        ``n_nodes`` (total leaf nodes = 2^level).
    """
    from .dwtfn import _wavelet_filter

    x = np.asarray(x, dtype=float).ravel()
    lo, hi = _wavelet_filter(wavelet)
    tree = {(0, 0): x}
    for j in range(level):
        for n in range(2**j):
            node = tree.get((j, n))
            if node is None or len(node) < len(lo):
                continue
            ca = np.convolve(node, lo, mode="full")[::2]
            cd = np.convolve(node, hi, mode="full")[::2]
            tree[(j + 1, 2 * n)] = ca
            tree[(j + 1, 2 * n + 1)] = cd
    n_leaves = 2**level
    return DescriptiveResult(
        name="wavelet_packets",
        value=float(n_leaves),
        extra={"tree": tree, "n_nodes": n_leaves, "level": level},
    )


wvpkt = wavelet_packets


def cheatsheet() -> str:
    return "wavelet_packets({}) -> Wavelet packet decomposition (full tree)."
