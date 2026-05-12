"""Simulated power of the Wilcoxon signed-rank test (Gibbons Ch 5.7.3).

Monte-Carlo power: draw ``nsim`` samples of size n from the
alternative (default: Normal with mean shift = ``effect_size``,
sd = 1), run a two-sided Wilcoxon signed-rank test, and report
the rejection rate at level ``alpha``.

The observed sample ``x`` is used only to set the sample size n;
the simulation distribution is independent of x by design (this
is the *power function*, not a post-hoc test).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["wilcoxon_power"]


def wilcoxon_power(
    x,
    effect_size: float = 0.5,
    alpha: float = 0.05,
    nsim: int = 2000,
    seed: int | None = 0,
):
    """Monte-Carlo power of the Wilcoxon signed-rank test.

    Parameters
    ----------
    x : array-like
        Sample (only ``len(x)`` is used).
    effect_size : float
        Location shift under H1 (Normal(effect_size, 1)).
    alpha : float
        Test level.
    nsim : int
        Simulation replicates.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    RichResult with payload:
        statistic : empirical power (rejection rate at H1)
        n         : sample size
        effect_size, alpha, nsim : echoed
        se        : Monte Carlo standard error of the power estimate
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 5:
        return RichResult(payload={
            "statistic": np.nan, "n": n, "effect_size": float(effect_size),
            "alpha": float(alpha), "nsim": int(nsim), "se": np.nan,
            "method": "Wilcoxon signed-rank power (Monte Carlo)",
        })
    rng = np.random.default_rng(seed)
    rejections = 0
    # zero-mean=0 method to handle small-sample exact for n<=25, else approx
    for _ in range(int(nsim)):
        sample = rng.normal(loc=effect_size, scale=1.0, size=n)
        try:
            res = stats.wilcoxon(sample, alternative="two-sided",
                                 zero_method="wilcox", correction=False)
            if res.pvalue < alpha:
                rejections += 1
        except ValueError:
            # all-zeros corner case
            pass
    power = rejections / float(nsim)
    se = float(np.sqrt(power * (1.0 - power) / nsim))
    return RichResult(payload={
        "statistic": float(power),
        "n": n,
        "effect_size": float(effect_size),
        "alpha": float(alpha),
        "nsim": int(nsim),
        "se": se,
        "method": "Wilcoxon signed-rank power (Monte Carlo)",
    })


def cheatsheet():
    return "wsrpw: Monte-Carlo power of Wilcoxon signed-rank test"


# CANONICAL TEST
# >>> wilcoxon_power(np.zeros(20), effect_size=0.5, alpha=0.05, nsim=2000)
# Expected power ≈ 0.55-0.65 for n=20, delta=0.5, sigma=1
