# morie.fn -- function file (hadesllm/morie)
"""Empirical process indexed by function class (Kosorok 2008, Ch 2).

G_n(f) = sqrt(n) * (P_n - P)(f).  If P is unknown (the usual case),
P(f) is estimated by an extra anchor argument or by f.mean(); the
empirical process value returned is then a centred and rescaled
sum, equal to (1/sqrt(n)) * sum_i (f(X_i) - P_n(f)) which is 0 by
construction.  The useful object is the *standardised* empirical
process Z_n(f) = sqrt(n)*(P_n(f) - mu0) under a hypothesised mu0,
which is what we compute alongside the asymptotic SE sigma/sqrt(n).
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_empirical_process"]


def kosorok_empirical_process(x, f=None, mu0=0.0):
    """Empirical process G_n(f) = sqrt(n)*(P_n - P)(f).

    Parameters
    ----------
    x : array-like
        IID sample X_1, ..., X_n.
    f : callable or None
        Function f.  If None, f is the identity.
    mu0 : float
        Hypothesised population mean P(f) under H_0.

    Returns
    -------
    RichResult with keys: estimate, se, n, method.
        estimate : sqrt(n)*(P_n(f) - mu0), Donsker-CLT statistic.
        se       : empirical sqrt of Var_P_n(f), the
                   asymptotic standard deviation of G_n(f).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    fx = x if f is None else np.asarray([f(xi) for xi in x], dtype=float)
    pn = float(np.mean(fx))
    estimate = float(np.sqrt(n) * (pn - mu0))
    se = float(np.std(fx, ddof=1)) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": estimate, "se": se, "n": n,
        "method": "Empirical process G_n(f) = sqrt(n)(P_n - P)(f)",
    })


def cheatsheet():
    return "ksr01: Empirical process G_n(f) = sqrt(n)(P_n - P)(f)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xs = rng.normal(size=200)
    r = kosorok_empirical_process(xs, mu0=0.0)
    print(r.estimate, r.se, r.n)
