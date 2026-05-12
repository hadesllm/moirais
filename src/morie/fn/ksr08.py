# morie.fn — function file (hadesllm/morie)
"""Multiplier bootstrap for Z-estimators (Kosorok 2008, Ch 10).

G_n^xi(f) = n^{-1/2} sum_i xi_i (f(X_i) - P_n f), with xi_i ~ N(0,1)
IID independent of the data.  Conditional on the data this Gaussian
multiplier process converges to the same limit as G_n.  We compute
the Monte-Carlo standard deviation of the multiplier statistic
applied to f(x)=x.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_multiplier_bootstrap"]


def kosorok_multiplier_bootstrap(x, B=1000, seed=0, deterministic_seed: int | None = None):
    """Gaussian-multiplier bootstrap SE of sqrt(n) P_n(id).

    Parameters
    ----------
    x : array-like.
    B : int, multiplier replications.
    seed : int.
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged: the user-supplied ``seed`` drives a fresh
        :class:`numpy.random.Generator`.

    Returns
    -------
    RichResult with: estimate (sample mean of G_n^xi), se (sd of
    G_n^xi across B), n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("ksr08", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    pn = float(x.mean())
    centred = x - pn
    xi = rng.normal(size=(B, n))
    g_xi = (xi @ centred) / np.sqrt(n)
    return RichResult(payload={
        "estimate": float(g_xi.mean()),
        "se":       float(g_xi.std(ddof=1)),
        "n":        n,
        "method":   "Multiplier bootstrap G_n^xi = n^{-1/2} sum xi (f-Pf)",
    })


def cheatsheet():
    return "ksr08: multiplier bootstrap G_n^xi = n^{-1/2} sum xi (f-Pf)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(kosorok_multiplier_bootstrap(rng.normal(size=200), B=500, seed=42))
