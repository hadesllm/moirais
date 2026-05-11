# morie.fn — function file (hadesllm/morie)
"""Ljung-Box test with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import chi2


def ljbox(residuals: Union[Sequence, np.ndarray], lags: int = 10):
    """Ljung-Box Q-statistic for serial correlation in residuals."""
    from ._richresult import hypothesis_test_result
    e = np.asarray(residuals, dtype=float)
    n = e.size
    if lags < 1 or lags >= n:
        raise ValueError(f"require 1 <= lags < n; got lags={lags}, n={n}.")
    e_c = e - e.mean()
    g0 = np.sum(e_c ** 2)
    if g0 == 0:
        raise ValueError("zero variance.")
    rho = np.array([np.sum(e_c[:n - k] * e_c[k:]) / g0 for k in range(1, lags + 1)])
    Q = float(n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1))))
    return hypothesis_test_result(
        test_name="Ljung-Box test for residual autocorrelation",
        statistic=Q, df=lags, pvalue=float(1 - chi2.cdf(Q, lags)),
        extra_summary=[
            ("Lags tested", lags),
            ("n residuals", n),
            ("Max |rho| up to lag", float(np.max(np.abs(rho)))),
        ],
    )
