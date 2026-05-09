"""Besag's L function (transformed K)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def l_function(K_values, r_values):
    """Compute L(r) = sqrt(K(r)/pi) - r (Besag's L function).

    L(r) = 0 under CSR, > 0 clustering, < 0 regularity.

    .. epigraph:: "Expelliarmus!" -- Harry Potter, Harry Potter

    Parameters
    ----------
    K_values : array_like
        Ripley's K values.
    r_values : array_like
        Corresponding distances.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    K = np.asarray(K_values, dtype=np.float64)
    r = np.asarray(r_values, dtype=np.float64)

    K_safe = np.maximum(K, 0)
    L = np.sqrt(K_safe / np.pi) - r

    return DescriptiveResult(
        name="l_function",
        value=float(np.max(np.abs(L))),
        extra={
            "r_values": r.tolist(),
            "L_values": L.tolist(),
            "max_abs_L": float(np.max(np.abs(L))),
            "max_L_distance": float(r[np.argmax(np.abs(L))]),
        },
    )


sglfn = l_function


def cheatsheet() -> str:
    return "l_function({}) -> Besag's L function (transformed K)."
