# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Benchmark dose (BMD/BMDL) calculation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ._containers import ESRes


def benchmark_dose(doses: np.ndarray | list, responses: np.ndarray | list, cdf=None, *, bmr: float = 0.10, alpha: float = 0.05) -> ESRes:
    """
    Estimate the Benchmark Dose (BMD) and its lower confidence limit (BMDL).

    Fits a probit dose-response model and finds the dose yielding a
    specified benchmark response level above background.

    Parameters
    ----------
    doses : array-like
        Dose levels (non-negative).
    responses : array-like
        Response proportions (0-1) at each dose.
    bmr : float
        Benchmark response level (default 10%).
    alpha : float
        Significance level for BMDL.

    Returns
    -------
    ESRes
        estimate = BMD, extra has 'bmdl', 'background_rate'.

    References
    ----------
    Crump, K. S. (1984). A new method for determining allowable daily
    intakes. *Fundam Appl Toxicol*, 4(5), 854-871.
    """
    d = np.asarray(doses, dtype=float)
    r = np.asarray(responses, dtype=float)
    if len(d) != len(r):
        raise ValueError("doses and responses must match.")
    if len(d) < 3:
        raise ValueError("Need at least 3 dose-response pairs.")

    from scipy.optimize import curve_fit

    def probit_model(x, a, b):
        return norm.cdf(a + b * x)

    popt, pcov = curve_fit(probit_model, d, r, p0=[0.0, 0.01], maxfev=5000)
    a, b = popt

    p0 = norm.cdf(a)
    target_p = p0 + bmr * (1 - p0)

    try:
        bmd = brentq(lambda x: norm.cdf(a + b * x) - target_p, 0, d.max() * 10)
    except ValueError:
        bmd = float("nan")

    se_b = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.001
    z = norm.ppf(1 - alpha)
    b_lower = b - z * se_b
    if abs(b_lower) > 1e-10:
        try:
            bmdl = brentq(lambda x: norm.cdf(a + b_lower * x) - target_p, 0, d.max() * 10)
        except ValueError:
            bmdl = float("nan")
    else:
        bmdl = float("nan")

    return ESRes(
        measure="benchmark_dose",
        estimate=float(bmd),
        ci_lower=float(bmdl),
        n=len(d),
        extra={"bmdl": float(bmdl), "background_rate": float(p0), "bmr": bmr},
    )


bench = benchmark_dose


def cheatsheet() -> str:
    return "benchmark_dose({}) -> Benchmark dose (BMD/BMDL) calculation."
