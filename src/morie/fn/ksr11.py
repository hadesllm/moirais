# morie.fn -- function file (hadesllm/morie)
"""Efficient score for the linear regression model (Kosorok 2008, Ch 6).

In the semiparametric model Y = beta X + eps with eps ind. of X and
E[eps]=0, the efficient score for beta is
    S_eff(X, Y) = (Y - E[Y|X]) (X - E[X]) / sigma^2.
Empirically: S_eff_i = (Y_i - Y_bar - beta_hat (X_i - X_bar)) (X_i - X_bar) / s_resid^2.
We return the empirical-mean efficient score (which should be 0 at
beta_hat) and its information-bound denominator s_resid^2 reported
as the SE component.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_efficient_score"]


def kosorok_efficient_score(x, y):
    """Empirical efficient score for OLS beta with unknown residual variance.

    Parameters
    ----------
    x, y : array-like.

    Returns
    -------
    RichResult with keys estimate (mean efficient score), se (residual
    sd as the nuisance-projection norm), n, method.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xc = x - x.mean(); yc = y - y.mean()
    beta_hat = float((xc @ yc) / (xc @ xc))
    resid = yc - beta_hat * xc
    sigma2 = float((resid ** 2).sum() / (n - 2)) if n > 2 else float("nan")
    s_eff = (resid * xc) / sigma2
    return RichResult(payload={
        "estimate": float(s_eff.mean()),
        "se":       float(np.sqrt(sigma2)),
        "n":        n,
        "method":   "Efficient score for OLS beta: (resid)(x-xbar)/sigma^2",
    })


def cheatsheet():
    return "ksr11: efficient score for OLS beta in linear model"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xs = rng.normal(size=200)
    ys = 1.5 * xs + rng.normal(size=200)
    print(kosorok_efficient_score(xs, ys))
