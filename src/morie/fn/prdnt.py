# morie.fn -- function file (hadesllm/morie)
"""Bootstrap prediction intervals for time series forecasts."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ["prdnt", "prediction_intervals"]


def prediction_intervals(
    y,
    fitted,
    *,
    alpha: float = 0.05,
    method: str = "residual",
    n_boot: int = 1000,
    seed: int = 42,
) -> DescriptiveResult:
    """Bootstrap prediction intervals for one-step-ahead forecasts.

    Constructs prediction intervals by resampling in-sample residuals
    (``y - fitted``) and adding them to the last fitted value.  Two
    resampling strategies are supported:

    * ``'residual'`` -- draws a single residual uniformly from the in-sample
      residual pool and adds it to ``fitted[-1]``.  Appropriate when residuals
      are approximately i.i.d.
    * ``'block'`` -- block bootstrap with block length
      ``ceil(n^{1/3})`` to preserve short-range serial dependence in the
      residuals (Carlstein 1986).

    Parameters
    ----------
    y : array-like
        Observed time series (n,).
    fitted : array-like
        In-sample fitted values (n,) -- must be same length as *y*.
    alpha : float
        Significance level for the (1−alpha)×100% prediction interval.
        Default 0.05 -> 95% PI.
    method : {'residual', 'block'}
        Resampling strategy.  Default ``'residual'``.
    n_boot : int
        Number of bootstrap replicates.  Default 1000.
    seed : int
        Random seed for reproducibility.  Default 42.

    Returns
    -------
    DescriptiveResult
        value: float -- point forecast (mean of bootstrap distribution).
        extra keys:
          'lower'       : float -- lower prediction bound.
          'upper'       : float -- upper prediction bound.
          'se'          : float -- bootstrap standard error.
          'forecast'    : float -- same as value.
          'alpha'       : float.
          'method'      : str.
          'n_boot'      : int.
          'residual_std': float -- in-sample residual standard deviation.

    Raises
    ------
    ValueError
        If y and fitted have different lengths or alpha is out of range.

    References
    ----------
    Thombs L.A. & Schucany W.R. (1990). Bootstrap prediction intervals for
    autoregression.
    Journal of the American Statistical Association, 85(410), 486-492.

    Carlstein E. (1986). The use of subseries values for estimating the
    variance of a general statistic from a stationary sequence.
    Annals of Statistics, 14(3), 1171-1179.
    """
    y = np.asarray(y, dtype=float).ravel()
    fitted = np.asarray(fitted, dtype=float).ravel()
    if len(y) != len(fitted):
        raise ValueError(
            f"y and fitted must have the same length; got {len(y)} and {len(fitted)}."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if method not in ("residual", "block"):
        raise ValueError(f"method must be 'residual' or 'block', got '{method}'.")

    n = len(y)
    resid = y - fitted
    resid_std = float(np.std(resid, ddof=1)) if n > 1 else float(np.abs(resid[0]))
    last_fit = float(fitted[-1])

    rng = np.random.default_rng(seed)

    if method == "residual":
        # i.i.d. residual resampling.
        sampled = rng.choice(resid, size=n_boot, replace=True)
        boot_forecasts = last_fit + sampled

    else:
        # Block bootstrap: preserve serial correlation in residuals.
        block_len = max(1, int(np.ceil(n ** (1.0 / 3.0))))
        starts = np.arange(0, n - block_len + 1)
        boot_forecasts = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            # Draw one block at random and take its first element as the
            # one-step-ahead residual.
            s = int(rng.choice(starts))
            boot_forecasts[i] = last_fit + resid[s]

    lower = float(np.quantile(boot_forecasts, alpha / 2.0))
    upper = float(np.quantile(boot_forecasts, 1.0 - alpha / 2.0))
    forecast = float(np.mean(boot_forecasts))
    se = float(np.std(boot_forecasts, ddof=1))

    return DescriptiveResult(
        name="prediction_intervals",
        value=forecast,
        extra={
            "lower": lower,
            "upper": upper,
            "se": se,
            "forecast": forecast,
            "alpha": float(alpha),
            "method": method,
            "n_boot": int(n_boot),
            "residual_std": resid_std,
        },
    )


prdnt = prediction_intervals


def cheatsheet() -> str:
    return "prediction_intervals(y, fitted, alpha=0.05) -> Bootstrap prediction intervals."
