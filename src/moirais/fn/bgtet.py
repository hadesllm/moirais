# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Breusch-Godfrey test for serial correlation."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def bg_test(y: np.ndarray, X: np.ndarray, order: int = 1, cdf=None) -> DescriptiveResult:
    """
    Breusch-Godfrey LM test for serial correlation up to order *p*.

    :param y: (n,) dependent variable.
    :param X: (n, k) regressor matrix.
    :param order: Order of serial correlation to test. Default 1.
    :return: DescriptiveResult with LM statistic and p-value.
    :raises ValueError: If insufficient data.

    References
    ----------
    Breusch T.S. (1978). Testing for autocorrelation in dynamic linear
    models. *Australian Economic Papers*, 17(31), 334-355.

    Godfrey L.G. (1978). Testing against general autoregressive and
    moving average error models. *Econometrica*, 46(6), 1293-1301.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if n < k + order + 2:
        raise ValueError(f"Need at least {k + order + 2} observations, got {n}.")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    Z = np.zeros((n, order))
    for lag in range(1, order + 1):
        Z[lag:, lag - 1] = resid[:-lag]
    X_aux = np.column_stack([X, Z])
    beta_aux = np.linalg.lstsq(X_aux, resid, rcond=None)[0]
    fitted_aux = X_aux @ beta_aux
    ss_res = float(np.sum((resid - fitted_aux) ** 2))
    ss_tot = float(np.sum(resid ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    lm_stat = n * r2
    p_val = 1 - stats.chi2.cdf(lm_stat, order)
    return DescriptiveResult(
        name="bg_test",
        value=float(lm_stat),
        extra={
            "lm_statistic": float(lm_stat),
            "p_value": float(p_val),
            "order": order,
            "r_squared_aux": float(r2),
            "n": n,
        },
    )


bgtet = bg_test


def cheatsheet() -> str:
    return "bg_test({}) -> Breusch-Godfrey serial correlation LM test."
