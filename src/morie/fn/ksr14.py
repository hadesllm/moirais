# morie.fn -- function file (hadesllm/morie)
"""Profile likelihood for the linear-regression slope (Kosorok 2008, Ch 7).

In Y = beta X + eps with eps ~ N(0, sigma^2), profiling out sigma^2
gives  l_p(beta) = -n/2 * log( SSR(beta)/n ) - n/2 (1 + log(2pi)),
maximised at the OLS slope.  Returns beta_hat and observed-information SE.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_profile_likelihood"]


def kosorok_profile_likelihood(x, y):
    """Profile-likelihood estimate of beta in Y = beta X + eps."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xc = x - x.mean(); yc = y - y.mean()
    Sxx = float(xc @ xc)
    beta = float((xc @ yc) / Sxx)
    resid = yc - beta * xc
    sigma2 = float((resid ** 2).sum() / (n - 2)) if n > 2 else float("nan")
    se = float(np.sqrt(sigma2 / Sxx))
    return RichResult(payload={
        "estimate": beta, "se": se, "n": n,
        "method": "Profile likelihood for OLS slope (eta=sigma^2 profiled)",
    })


def cheatsheet():
    return "ksr14: profile-likelihood OLS slope"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xs = rng.normal(size=200)
    ys = 1.5 * xs + rng.normal(size=200)
    print(kosorok_profile_likelihood(xs, ys))
