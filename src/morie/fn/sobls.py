r"""Sobol quasi-random low-discrepancy sequence (Sobol 1967).

Thin wrapper around ``scipy.stats.qmc.Sobol`` (deterministic for fixed
seed).  Returns the QMC point set in [0,1]^d plus an optional QMC
estimate ``\\hat\\theta = (1/N) sum f(x_i)`` for integrand ``f``.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["sobol_sequence"]


def sobol_sequence(x=None, N: int = 128, d: int = 1, f=None,
                   scramble: bool = True, seed: int = 42):
    """Sobol quasi-random sequence (low-discrepancy).

    Parameters
    ----------
    x : ignored / optional (accepted for spec parity)
    N : int
        Number of points; recommended power of 2.  Default 128.
    d : int
        Dimensionality.  Default 1.
    f : callable, optional
        Integrand on [0,1]^d → scalar.
    scramble : bool
        Owen-scrambling (default True) for randomised QMC.
    seed : int

    Returns
    -------
    RichResult: sample, [estimate, se], N, d, method.
    """
    try:
        from scipy.stats import qmc
    except Exception as e:  # pragma: no cover
        return RichResult(payload={"error": f"scipy.stats.qmc unavailable: {e}",
                                   "method": "Sobol (Sobol 1967)"})
    if x is not None:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            N = x_arr.size
        elif x_arr.ndim == 2:
            N, d = x_arr.shape
    eng = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    # qmc.Sobol expects log2 N for the balanced sequence; fall back to .random
    m = int(np.ceil(np.log2(max(1, N))))
    sample = eng.random_base2(m)[:N]
    payload = {"sample": sample, "N": int(N), "d": int(d),
               "method": "Sobol QMC (Sobol 1967)"}
    if f is not None:
        try:
            fv = np.array([f(sample[i]) for i in range(N)], dtype=float)
        except Exception:
            fv = np.asarray(f(sample), dtype=float)
        payload["estimate"] = float(fv.mean())
        payload["se"] = float(fv.std(ddof=1) / np.sqrt(N))
    return RichResult(payload=payload)


# CANONICAL TEST
# >>> res = sobol_sequence(N=128, d=2, f=lambda u: u[0]*u[1], seed=0)
# >>> # E_U[U1*U2] = 0.25
# >>> assert abs(res["estimate"] - 0.25) < 0.05


def cheatsheet():
    return "sobls(N=128, d=1, f=None): Sobol QMC points + integral estimate."
