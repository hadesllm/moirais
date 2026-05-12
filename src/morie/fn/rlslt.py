# morie.fn -- function file (hadesllm/morie)
"""RLS lattice adaptive filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def rls_lattice_filter_fn(
    x: np.ndarray,
    d: np.ndarray,
    order: int = 8,
    lam: float = 0.99,
) -> SignalResult:
    """Apply RLS lattice adaptive filter.

    :param x: 1-D input signal.
    :param d: 1-D desired signal.
    :param order: Filter order (default 8).
    :param lam: Forgetting factor (default 0.99).
    :return: SignalResult with filtered output.
    """
    from morie._adaptive import rls_lattice_filter

    x = np.asarray(x, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    y, e = rls_lattice_filter(x, d, order=order, lam=lam)
    return SignalResult(
        name="rls_lattice_filter",
        filtered=y,
        fs=1.0,
        n_samples=len(y),
        extra={"error": e, "order": order},
    )


rlslt = rls_lattice_filter_fn


def cheatsheet() -> str:
    return "rls_lattice_filter_fn({}) -> RLS lattice adaptive filter."
