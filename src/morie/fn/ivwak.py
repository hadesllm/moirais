# morie.fn -- function file (hadesllm/morie)
"""Weak instrument diagnostics for IV estimation."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def iv_weak_test(endogenous, instrument, covariates=None, cdf=None):
    """
    Test for weak instruments using first-stage F-statistic.

    Stock-Yogo critical values: F < 10 suggests weak instrument.

    :param endogenous: (n,) endogenous regressor.
    :param instrument: (n,) or (n,k) excluded instruments.
    :param covariates: (n,p) included exogenous covariates.
    :return: DescriptiveResult with F-stat, partial R², Cragg-Donald stat.

    References
    ----------
    Stock JH, Yogo M (2005). Testing for Weak Instruments in Linear IV
    Regression. In: Andrews DWK (ed) Identification and Inference for
    Econometric Models. Cambridge University Press.
    """
    y = np.asarray(endogenous, dtype=np.float64).ravel()
    Z = np.asarray(instrument, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    n, l = Z.shape

    if covariates is not None:
        W = np.column_stack([np.ones(n), np.asarray(covariates)])
    else:
        W = np.ones((n, 1))

    Mw = np.eye(n) - W @ np.linalg.pinv(W)
    y_res = Mw @ y
    Z_res = Mw @ Z

    X_full = np.column_stack([W, Z])
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    ss_full = np.sum((y - X_full @ beta_full) ** 2)
    beta_red = np.linalg.lstsq(W, y, rcond=None)[0]
    ss_red = np.sum((y - W @ beta_red) ** 2)

    df1 = l
    df2 = n - W.shape[1] - l
    f_stat = ((ss_red - ss_full) / df1) / (ss_full / df2) if df2 > 0 else 0.0
    f_pval = 1 - stats.f.cdf(f_stat, df1, df2) if df2 > 0 else 1.0

    ss_total_res = np.sum(y_res**2)
    partial_r2 = 1 - ss_full / ss_red if ss_red > 0 else 0.0
    weak = f_stat < 10

    return DescriptiveResult(
        name="iv_weak_test",
        value=float(f_stat),
        extra={
            "f_statistic": float(f_stat),
            "f_pvalue": float(f_pval),
            "partial_r_squared": float(partial_r2),
            "n_instruments": l,
            "weak_instrument": bool(weak),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "iv_weak_test({}) -> Weak instrument diagnostics for IV estimation."
