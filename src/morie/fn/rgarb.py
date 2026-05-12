# morie.fn -- function file (hadesllm/morie)
"""AR(p) model via Burg's recursion -- Rangayyan Ch 4."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_ar_burg"]


def rangayyan_ar_burg(x, order=10):
    """Burg's method for autoregressive (AR) coefficients.

    Estimates AR(p) coefficients ``a_1, …, a_p`` and innovation variance
    ``σ²`` from data ``x`` by minimising the sum of forward+backward
    prediction error energies (always yields a stable model).

    Sign convention: ``x[n] = -Σ a_k x[n-k] + e[n]``.

    Parameters
    ----------
    x : array-like
    order : int
        AR model order.

    Returns
    -------
    RichResult with keys ``ar_coeffs``, ``variance``, ``order``, ``reflection``.

    References
    ----------
    Rangayyan Ch 4; Burg (1975); Marple (1987) §8.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    p = int(order)
    if p < 1 or p >= N:
        raise ValueError("order must be 1 <= p < len(x)")

    f = x.copy()
    b = x.copy()
    a = np.zeros(p + 1)
    a[0] = 1.0
    var = float(np.mean(x ** 2))
    k = np.zeros(p)
    for m in range(p):
        num = -2.0 * np.sum(f[m + 1:N] * b[m:N - 1])
        den = np.sum(f[m + 1:N] ** 2) + np.sum(b[m:N - 1] ** 2)
        km = (num / den) if den > 0 else 0.0
        k[m] = km
        new_a = a.copy()
        for i in range(1, m + 2):
            new_a[i] = a[i] + km * a[m + 1 - i]
        a = new_a
        f_new = f[m + 1:N] + km * b[m:N - 1]
        b_new = b[m:N - 1] + km * f[m + 1:N]
        f[m + 1:N] = f_new
        b[m + 1:N] = b_new
        var = var * (1.0 - km * km)

    res = RichResult(
        title=f"AR({p}) Burg model",
        summary_lines=[
            ("Order", p),
            ("Innovation variance σ²", float(var)),
            ("First reflection k_1", float(k[0])),
        ],
        interpretation=f"Stable AR({p}) fit; residual variance {var:.4g}.",
        payload={"ar_coeffs": a[1:], "variance": float(var),
                 "order": p, "reflection": k},
    )
    return with_describe_pointer(res, "rgarb")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = rangayyan_ar_burg(x, order=4)
# >>> r["ar_coeffs"].shape
# (4,)


def cheatsheet():
    return "rgarb: AR(p) coefficients via Burg method -- Rangayyan Ch 4"
