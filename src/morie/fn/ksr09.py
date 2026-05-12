# morie.fn -- function file (hadesllm/morie)
"""Z-estimator asymptotic distribution (Kosorok 2008, Ch 5).

A Z-estimator theta_n solves P_n psi(.; theta) = 0.  Under regularity,
sqrt(n)(theta_n - theta_0) -> N(0, V) with sandwich variance
    V = A^{-1} B A^{-T}, A = P psi_dot(theta_0), B = P psi(theta_0)^{2}.
With y omitted we solve psi(x; theta) = x - theta = 0 (the sample
mean).  With y supplied we solve psi(x, y; beta) = x(y - beta x) = 0
(centred OLS slope), both with sandwich SE.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_z_estimator"]


def kosorok_z_estimator(x, y=None):
    """Z-estimator: location (one-sample) or OLS slope (two-sample).

    Parameters
    ----------
    x : array-like.
    y : array-like or None.

    Returns
    -------
    RichResult with keys estimate, se, n, method.
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        n = len(x)
        theta = float(x.mean())
        psi = x - theta
        v = float((psi ** 2).mean())
        se = float(np.sqrt(v / n)) if n > 0 else float("nan")
        method = "Z-estimator: psi(x;theta) = x - theta"
        est = theta
    else:
        y = np.asarray(y, dtype=float)
        n = len(x)
        xc = x - x.mean(); yc = y - y.mean()
        beta = float((xc @ yc) / (xc @ xc))
        resid = yc - beta * xc
        A = float((xc ** 2).mean())
        B = float(((xc ** 2) * (resid ** 2)).mean())
        se = float(np.sqrt(B / (A ** 2) / n))
        method = "Z-estimator: psi(x,y;beta) = x(y - beta x)"
        est = beta
    return RichResult(payload={
        "estimate": est, "se": se, "n": n, "method": method,
    })


def cheatsheet():
    return "ksr09: Z-estimator with sandwich SE"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xs = rng.normal(size=200)
    ys = 1.5 * xs + rng.normal(size=200)
    print(kosorok_z_estimator(xs, ys))
