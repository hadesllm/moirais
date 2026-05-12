# morie.fn -- function file (hadesllm/morie)
"""Donsker-class verification via the bracketing integral (Kosorok 2008, Ch 2).

A class F is P-Donsker if the L_2(P) bracketing entropy integral
    J_[](1, F, L_2(P)) = int_0^1 sqrt(log N_[](e, F, L_2(P))) de
is finite (Theorem 2.5.2 in Kosorok 2008).  For the univariate
indicator class F = {1{X<=t} : t in R} one has N_[](e, F, L_2(P))
<= 2/e^2 (van der Vaart & Wellner, Ex. 2.5.4), giving a finite
entropy integral.  This callable returns that integral as a
quantitative Donsker-class diagnostic.
"""
import numpy as np
from scipy import integrate
from ._richresult import RichResult

__all__ = ["kosorok_donsker_class"]


def kosorok_donsker_class(x):
    """Donsker-class diagnostic for F = {1{X<=t} : t in R}.

    Parameters
    ----------
    x : array-like (unused, retained for API parity).

    Returns
    -------
    RichResult with: estimate (bracketing integral J_[](1)), n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    integrand = lambda e: np.sqrt(np.log(2.0) - 2.0 * np.log(e))
    j, _ = integrate.quad(integrand, 1e-8, 1.0, limit=200)
    return RichResult(payload={
        "estimate": float(j),
        "n": n,
        "method": "Bracketing-integral Donsker verification (indicator class)",
    })


def cheatsheet():
    return "ksr02: Donsker class via bracketing-integral J_[](1)"


# CANONICAL TEST
if __name__ == "__main__":
    print(kosorok_donsker_class(np.arange(10)).estimate)
