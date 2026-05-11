# morie.fn — function file (hadesllm/morie)
"""Durbin-Watson test statistic for autocorrelation."""

import numpy as np

from ._containers import DescriptiveResult


def durbin_watson(residuals: np.ndarray) -> DescriptiveResult:
    """
    Durbin-Watson statistic for first-order autocorrelation.

    .. math::

        DW = \\frac{\\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\\sum_{t=1}^{n} e_t^2}

    Values near 2 indicate no autocorrelation, near 0 positive, near 4
    negative.

    :param residuals: 1-D array of OLS residuals.
    :return: DescriptiveResult with DW statistic.
    :raises ValueError: If fewer than 3 observations.

    References
    ----------
    Durbin J. & Watson G.S. (1950). Testing for serial correlation in
    least squares regression I. *Biometrika*, 37(3-4), 409-428.
    """
    e = np.asarray(residuals, dtype=float).ravel()
    n = len(e)
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}.")
    ss = float(np.sum(e ** 2))
    if ss < 1e-15:
        return DescriptiveResult(name="durbin_watson", value=2.0, extra={"dw": 2.0, "n": n})
    dw = float(np.sum(np.diff(e) ** 2) / ss)
    rho1 = 1 - dw / 2
    return DescriptiveResult(
        name="durbin_watson",
        value=dw,
        extra={
            "dw": dw,
            "approx_rho1": float(rho1),
            "n": n,
        },
    )


dwtst = durbin_watson


def cheatsheet() -> str:
    return "durbin_watson({}) -> Durbin-Watson test statistic."
