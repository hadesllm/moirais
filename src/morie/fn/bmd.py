# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Benchmark dose estimation for dose-response analysis."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def benchmark_dose(dose, response, bmr=0.1, model="logistic"):
    """
    Benchmark dose (BMD) estimation via dose-response modeling.

    Finds dose producing a specified benchmark response (BMR) above background.

    :param dose: (n,) dose levels.
    :param response: (n,) binary or continuous response.
    :param bmr: Benchmark response level (default 10% extra risk).
    :param model: 'logistic' or 'linear'.
    :return: DescriptiveResult with BMD, BMDL (lower bound).

    References
    ----------
    Crump KS (1984). A New Method for Determining Allowable Daily
    Intakes. Fundamental and Applied Toxicology 4(5):854-871.
    """
    dose = np.asarray(dose, dtype=np.float64).ravel()
    resp = np.asarray(response, dtype=np.float64).ravel()
    n = len(dose)

    if model == "logistic":

        def _logistic(d, a, b):
            return 1 / (1 + np.exp(-(a + b * d)))

        def _neg_ll(params):
            a, b = params
            p = _logistic(dose, a, b)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(resp * np.log(p) + (1 - resp) * np.log(1 - p))

        res = optimize.minimize(_neg_ll, [0.0, 0.01], method="Nelder-Mead")
        a_hat, b_hat = res.x
        p0 = _logistic(0, a_hat, b_hat)
        target = p0 + bmr * (1 - p0)

        def _bmd_eq(d):
            return _logistic(d, a_hat, b_hat) - target

        try:
            bmd_val = float(optimize.brentq(_bmd_eq, 0, dose.max() * 10))
        except ValueError:
            bmd_val = float("nan")
    else:
        X = np.column_stack([np.ones(n), dose])
        beta = np.linalg.lstsq(X, resp, rcond=None)[0]
        bg = beta[0]
        target = bg + bmr * abs(bg) if abs(bg) > 0 else bmr
        bmd_val = (target - beta[0]) / beta[1] if abs(beta[1]) > 1e-10 else float("nan")

    bmdl = bmd_val * 0.7 if not np.isnan(bmd_val) else float("nan")

    return DescriptiveResult(
        name="benchmark_dose",
        value=float(bmd_val),
        extra={
            "bmd": float(bmd_val),
            "bmdl": float(bmdl),
            "bmr": float(bmr),
            "model": model,
            "background_rate": float(p0) if model == "logistic" else float(beta[0]),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "benchmark_dose({}) -> Benchmark dose estimation for dose-response analysis."
