# morie.fn — function file (hadesllm/morie)
"""Influence function for OLS beta (Kosorok 2008, Ch 7).

IF(x, y) = I_eff^{-1} S_eff(x, y) = (y - ybar - beta_hat (x - xbar))
* (x - xbar) / Var(X).  Returns the empirical mean of IF (0 by
construction).
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_influence_function"]


def kosorok_influence_function(x, y):
    """Empirical influence function of OLS beta-hat."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xc = x - x.mean(); yc = y - y.mean()
    var_x = float(xc @ xc / n)
    beta_hat = float((xc @ yc) / (xc @ xc))
    resid = yc - beta_hat * xc
    IF = (resid * xc) / var_x
    return RichResult(payload={
        "estimate": float(IF.mean()),
        "n":        n,
        "method":   "Influence function: (resid)(x-xbar)/Var(X)",
    })


def cheatsheet():
    return "ksr16: influence function of OLS beta-hat"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xs = rng.normal(size=200)
    ys = 1.5 * xs + rng.normal(size=200)
    print(kosorok_influence_function(xs, ys))
