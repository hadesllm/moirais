# morie.fn -- function file (hadesllm/morie)
"""Bootstrap consistency for the empirical process (Kosorok 2008, Ch 10).

G_n^*(f) = sqrt(n)(P_n^* - P_n)(f) converges (conditionally on the
data) to the same Gaussian limit as G_n.  We Monte-Carlo it: draw B
multinomial(n; 1/n,...,1/n) weights, recompute the bootstrap mean,
and return the bootstrap sd as the SE of sqrt(n) P_n.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_bootstrap_empirical"]


def kosorok_bootstrap_empirical(x, B=1000, seed=0, deterministic_seed: int | None = None):
    """Nonparametric bootstrap SE for sqrt(n)*(P_n - P)(id).

    Parameters
    ----------
    x : array-like, IID sample.
    B : int, number of bootstrap replications.
    seed : int.
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged: the user-supplied ``seed`` drives a fresh
        :class:`numpy.random.Generator`.

    Returns
    -------
    RichResult with: estimate (centred bootstrap mean of G_n^*),
    se (bootstrap sd of sqrt(n) P_n^*), n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("ksr07", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = x[idx].mean(axis=1)
    pn = float(x.mean())
    gn_star = np.sqrt(n) * (boot_means - pn)
    return RichResult(payload={
        "estimate": float(gn_star.mean()),
        "se":       float(gn_star.std(ddof=1)),
        "n":        n,
        "method":   "Bootstrap G_n^*(f) = sqrt(n)(P_n^* - P_n)",
    })


def cheatsheet():
    return "ksr07: bootstrap empirical process G_n^* = sqrt(n)(P_n^* - P_n)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(kosorok_bootstrap_empirical(rng.normal(size=200), B=500, seed=42))
