# morie.fn -- function file (hadesllm/morie)
"""One-step efficient estimator (Kosorok 2008, Ch 7).

theta_tilde = theta_init + n^{-1} sum IF(X_i; theta_init).
Starting from theta_init = median(x) and IF(x; theta) = x - theta,
one-step recovers the mean.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_one_step_estimator"]


def kosorok_one_step_estimator(x, y=None):
    """One-step location estimator from median initial.

    Parameters
    ----------
    x : array-like.
    y : ignored.

    Returns
    -------
    RichResult with keys estimate, se, n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    theta_init = float(np.median(x))
    IF = x - theta_init
    update = float(IF.mean())
    theta_tilde = theta_init + update
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": theta_tilde, "se": se, "n": n,
        "method":   "One-step from median: theta + mean(x-theta)",
    })


def cheatsheet():
    return "ksr15: one-step efficient location estimator"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(kosorok_one_step_estimator(rng.normal(size=200)))
