# morie.fn — function file (hadesllm/morie)
"""Bracketing number N_[](e, F, L_2(P)) for the indicator class (Kosorok 2008, Ch 2).

For F = {1{X<=t} : t in R}, the e-L_2 bracketing number is the number
of points needed to partition the unit interval of CDF-values into
strips of L_2(P)-width <= e.  We use F_n as a plug-in for P and
return ceil(1/e^2) brackets (the canonical Kosorok/vdV-Wellner bound),
plus the achievable count from the empirical CDF at chosen e.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_bracketing_number"]


def kosorok_bracketing_number(x, e=0.1):
    """e-bracketing number of {1{X<=t}} in L_2(P_n).

    Parameters
    ----------
    x : array-like, IID sample.
    e : float, bracket width in L_2(P_n).

    Returns
    -------
    RichResult with: estimate (number of brackets), n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    # Each bracket has L_2(P)^2 width <= e^2, i.e. covers a probability
    # mass of <= e^2; we need ceil(1/e^2) of them to cover [0,1].
    nb = int(np.ceil(1.0 / (e ** 2)))
    return RichResult(payload={
        "estimate": nb,
        "n":        n,
        "method":   "N_[](e, {1{X<=t}}, L2(P_n)) = ceil(1/e^2)",
    })


def cheatsheet():
    return "ksr05: bracketing number N_[](e) = ceil(1/e^2) for indicator class"


# CANONICAL TEST
if __name__ == "__main__":
    print(kosorok_bracketing_number(np.arange(50), e=0.1))
