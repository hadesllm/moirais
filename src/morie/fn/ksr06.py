# morie.fn — function file (hadesllm/morie)
"""Maximal inequality bound for empirical processes (Kosorok 2008, Ch 2).

For a P-Donsker class F with envelope in L_2(P):
    E*[sup_{f in F} |G_n(f)|] <= J_[](theta_n, F, L_2(P)) * sigma_n
where theta_n = sup_F ||f||_{L_2(P)}.  We use the indicator class with
N_[](e) <= 2/e^2 and theta_n = 0.5 (the maximal L_2-radius of a
0/1-valued class), returning the numerical Kosorok/Dudley RHS.
"""
import numpy as np
from scipy import integrate
from ._richresult import RichResult

__all__ = ["kosorok_maximal_inequality"]


def kosorok_maximal_inequality(x):
    """E*[sup |G_n(f)|] upper bound for the indicator class.

    Parameters
    ----------
    x : array-like.

    Returns
    -------
    RichResult with keys estimate, n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    sigma_n = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    theta_n = 0.5
    integrand = lambda e: np.sqrt(np.log(2.0) - 2.0 * np.log(e))
    j, _ = integrate.quad(integrand, 1e-8, theta_n, limit=200)
    bound = float(j * sigma_n) if np.isfinite(sigma_n) else float("nan")
    return RichResult(payload={
        "estimate": bound,
        "n":        n,
        "method":   "Maximal-inequality RHS: J_[](theta_n) * sigma_n",
    })


def cheatsheet():
    return "ksr06: E[sup |G_n(f)|] <= J_[](theta_n) * sigma_n"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(kosorok_maximal_inequality(rng.normal(size=200)))
