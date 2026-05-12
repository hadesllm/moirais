# morie.fn — function file (hadesllm/morie)
"""Latin hypercube sampling (McKay, Beckman & Conover 1979).

Generates an N×d stratified sample on [0,1]^d:

* divide each dimension into N equal-probability strata
* place exactly one sample at a random location in each stratum
* independently permute strata across dimensions

When an integrand ``f`` is provided, returns the LHS Monte-Carlo estimator
of ``E_U[f(U)]`` with its sample SE.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["latin_hypercube"]


def latin_hypercube(x=None, N: int = 100, d: int = 1, f=None, seed: int = 42):
    """Latin hypercube sample (+ optional Monte-Carlo integration).

    Parameters
    ----------
    x : ignored / optional
        Accepted for spec parity; the sample is generated internally.
    N : int
        Sample size.  Default 100.
    d : int
        Dimensionality.  Default 1.
    f : callable, optional
        Integrand ``f(u) -> scalar`` where u has shape (d,) or (N,d).
    seed : int

    Returns
    -------
    RichResult: sample (N,d) array, estimate, se (if f given), N, d, method.
    """
    rng = np.random.default_rng(seed)
    if x is not None:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            N = x_arr.size
        elif x_arr.ndim == 2:
            N, d = x_arr.shape
    cut = np.arange(N) / N
    sample = np.empty((N, d), dtype=float)
    for j in range(d):
        u_j = cut + rng.uniform(0.0, 1.0 / N, size=N)
        sample[:, j] = rng.permutation(u_j)
    payload = {"sample": sample, "N": int(N), "d": int(d),
               "method": "Latin hypercube (McKay et al. 1979)"}
    if f is not None:
        try:
            fv = np.array([f(sample[i]) for i in range(N)], dtype=float)
        except Exception:
            fv = np.asarray(f(sample), dtype=float)
        payload["estimate"] = float(fv.mean())
        payload["se"] = float(fv.std(ddof=1) / np.sqrt(N))
    return RichResult(payload=payload)


# CANONICAL TEST
# >>> import numpy as np
# >>> res = latin_hypercube(N=1000, d=2, f=lambda u: u[0] + u[1], seed=0)
# >>> # E_U[U1+U2] = 1
# >>> assert abs(res["estimate"] - 1.0) < 0.05


def cheatsheet():
    return "latnh(N=100, d=1, f=None): Latin hypercube sample + MC estimate."
