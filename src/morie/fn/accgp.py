# morie.fn — function file (hadesllm/morie)
"""Genomic-prediction accuracy metrics."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["prediction_accuracy"]


def prediction_accuracy(y_true, y_pred):
    """Standard accuracy metrics for genomic-prediction results.

    Reports:
        - Pearson r = cor(y, y_hat)
        - Spearman rho
        - MSE = mean((y - y_hat)^2)
        - MSPE = MSE (used as alias in genomics literature)
        - RMSE = sqrt(MSE)
        - Calibration slope b = cov(y, y_hat) / var(y_hat)
        - Calibration intercept a = mean(y) - b*mean(y_hat)
        - Coefficient of determination R^2 = 1 - MSE/var(y)

    Parameters
    ----------
    y_true : array-like (n,)
    y_pred : array-like (n,)

    Returns
    -------
    RichResult with payload keys estimate (Pearson r), spearman, mse, rmse,
    r2, slope, intercept, n, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 2.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if n < 2:
        return RichResult(
            payload={"estimate": float("nan"), "n": n,
                     "method": "Genomic prediction accuracy (n<2)"},
            warnings=["Need n >= 2 for accuracy metrics"],
        )
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    var_y = float(np.var(y_true, ddof=1))
    r2 = 1.0 - mse / var_y if var_y > 0 else float("nan")
    # Pearson r
    sd_t = float(np.std(y_true, ddof=1))
    sd_p = float(np.std(y_pred, ddof=1))
    if sd_t > 0 and sd_p > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = float("nan")
    # Spearman
    rt = np.argsort(np.argsort(y_true))
    rp = np.argsort(np.argsort(y_pred))
    if np.std(rt) > 0 and np.std(rp) > 0:
        rho = float(np.corrcoef(rt, rp)[0, 1])
    else:
        rho = float("nan")
    # Calibration slope/intercept of y ~ a + b*y_pred
    if np.var(y_pred, ddof=1) > 0:
        slope = float(np.cov(y_true, y_pred, ddof=1)[0, 1]
                      / np.var(y_pred, ddof=1))
        intercept = float(np.mean(y_true) - slope * np.mean(y_pred))
    else:
        slope = float("nan"); intercept = float("nan")
    return RichResult(
        title="Genomic-prediction accuracy",
        summary_lines=[
            ("n", n),
            ("Pearson r", r),
            ("Spearman rho", rho),
            ("MSE", mse),
            ("RMSE", rmse),
            ("R^2", r2),
            ("calibration slope", slope),
            ("calibration intercept", intercept),
        ],
        payload={
            "estimate": r,
            "pearson_r": r,
            "spearman_rho": rho,
            "mse": mse,
            "mspe": mse,
            "rmse": rmse,
            "r2": r2,
            "slope": slope,
            "intercept": intercept,
            "n": n,
            "method": "Pearson r + Spearman rho + MSE/MSPE + calibration",
        },
    )


def cheatsheet():
    return "accgp: Genomic-prediction accuracy metrics"


# CANONICAL TEST
# y = np.array([1.0,2.0,3.0,4.0,5.0]); y_hat = np.array([1.1,1.9,3.2,3.8,5.1])
# r = prediction_accuracy(y, y_hat); r.pearson_r > 0.99; MSE small.
