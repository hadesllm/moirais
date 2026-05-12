# morie.fn -- function file (hadesllm/morie)
"""Empirical Mode Decomposition (standalone).

Distinct from hhtfn.py: this module exposes EMD as a standalone
decomposition without the Hilbert spectrum step, and includes
diagnostic sifting metrics.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ._containers import DescriptiveResult
def _count_zero_crossings(x: np.ndarray) -> int:
    """Count zero crossings in signal *x*."""
    return int(np.sum(np.diff(np.sign(x)) != 0))


def _count_extrema(x: np.ndarray) -> int:
    """Count total number of local extrema."""
    max_idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]
    min_idx = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0]
    return len(max_idx) + len(min_idx)


def emd(
    x: np.ndarray,
    *,
    max_imfs: int = 12,
    max_sift_iter: int = 300,
    sd_threshold: float = 0.05,
) -> DescriptiveResult:
    r"""Empirical Mode Decomposition.

    Decomposes a signal into a set of Intrinsic Mode Functions (IMFs)
    through an iterative sifting process.  Each IMF satisfies two
    conditions:

    1. The number of extrema and zero crossings differ by at most one.
    2. The local mean of the upper and lower envelopes is zero.

    The sifting stopping criterion uses the normalized squared
    difference:

    .. math::

        SD = \\frac{\\sum_t |h_{k-1}(t) - h_k(t)|^2}
             {\\sum_t h_{k-1}^2(t)}

    Parameters
    ----------
    x : array-like
        1-D input signal.
    max_imfs : int
        Maximum number of IMFs to extract (default 12).
    max_sift_iter : int
        Maximum sifting iterations per IMF (default 300).
    sd_threshold : float
        Sifting convergence threshold (default 0.05).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``imfs`` (list of arrays), ``residue``,
        ``n_imfs``, ``sift_counts`` (iterations per IMF),
        ``is_imf`` (bool per IMF -- satisfies IMF conditions).

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Huang, N.E. et al. (1998). The empirical mode decomposition and
    the Hilbert spectrum for nonlinear and non-stationary time series
    analysis. *Proc. R. Soc. Lond. A*, 454, 903--995.
    doi:10.1098/rspa.1998.0193
    """
    x = np.asarray(x, dtype=float).ravel()
    residue = x.copy()
    imfs = []
    sift_counts = []

    for _ in range(max_imfs):
        h = residue.copy()
        n_sifts = 0

        for s in range(max_sift_iter):
            n_sifts = s + 1
            t = np.arange(len(h))
            max_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
            min_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1

            if len(max_idx) < 2 or len(min_idx) < 2:
                break

            upper = CubicSpline(max_idx, h[max_idx], extrapolate=True)(t)
            lower = CubicSpline(min_idx, h[min_idx], extrapolate=True)(t)
            mean_env = (upper + lower) / 2.0

            prev = h.copy()
            h = h - mean_env

            sd = np.sum((prev - h) ** 2) / (np.sum(prev ** 2) + 1e-12)
            if sd < sd_threshold:
                break

        if np.max(np.abs(h)) < 1e-10:
            break

        imfs.append(h)
        sift_counts.append(n_sifts)
        residue = residue - h

        max_idx = np.where(
            (residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:])
        )[0] + 1
        min_idx = np.where(
            (residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:])
        )[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

    is_imf = []
    for imf in imfs:
        zc = _count_zero_crossings(imf)
        ne = _count_extrema(imf)
        is_imf.append(abs(ne - zc) <= 1)

    return DescriptiveResult(
        name="emd",
        value=float(len(imfs)),
        extra={
            "imfs": imfs,
            "residue": residue,
            "n_imfs": len(imfs),
            "sift_counts": sift_counts,
            "is_imf": is_imf,
        },
    )


emdsg = emd


def cheatsheet() -> str:
    return "emd({}) -> Empirical Mode Decomposition (standalone with diagnostics)."
