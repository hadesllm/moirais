# morie.fn — function file (hadesllm/morie)
"""Semiparametric information bound (Kosorok 2008, Ch 6).

I_eff = E[S_eff S_eff'].  For the linear regression model
Y = beta X + eps, this is
    I_eff = Var(X) / sigma^2,
the Fisher-information lower bound on Var(sqrt(n) beta_hat).
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_information_bound"]


def kosorok_information_bound(x, y):
    """Empirical I_eff = Var(X)/sigma^2 for linear model.

    Parameters
    ----------
    x, y : array-like.

    Returns
    -------
    RichResult with keys estimate (I_eff), n, method.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xc = x - x.mean(); yc = y - y.mean()
    beta_hat = float((xc @ yc) / (xc @ xc))
    resid = yc - beta_hat * xc
    sigma2 = float((resid ** 2).sum() / (n - 2)) if n > 2 else float("nan")
    var_x = float(xc @ xc / n)
    i_eff = float(var_x / sigma2) if sigma2 > 0 else float("nan")
    return RichResult(payload={
        "estimate": i_eff,
        "n":        n,
        "method":   "Information bound I_eff = Var(X)/sigma^2",
    })


def cheatsheet():
    return "ksr12: information bound I_eff for linear model"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xs = rng.normal(size=200)
    ys = 1.5 * xs + rng.normal(size=200)
    print(kosorok_information_bound(xs, ys))
