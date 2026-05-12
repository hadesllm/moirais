"""Value at Risk (VaR)."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def value_at_risk(
    returns: np.ndarray,
    alpha: float = 0.05,
    method: str = "historical",
) -> DescriptiveResult:
    r"""Value at Risk (VaR) at a given confidence level.

    Three methods:

    - **historical**: empirical quantile of the loss distribution.
    - **parametric**: assumes normal returns,
      :math:`\\text{VaR} = -(\\mu + z_\\alpha \\sigma)`.
    - **cornish-fisher**: adjusts the normal quantile for skewness
      and kurtosis.

    Parameters
    ----------
    returns : array-like
        Array of portfolio returns.
    alpha : float, default 0.05
        Significance level (e.g. 0.05 for 95% VaR).
    method : str, default "historical"
        One of ``"historical"``, ``"parametric"``, ``"cornish-fisher"``.

    Returns
    -------
    DescriptiveResult
        ``value`` is the VaR (positive number = loss magnitude).
        ``extra`` has ``alpha``, ``method``, ``n_obs``.

    Raises
    ------
    ValueError
        If alpha out of (0, 1) or unknown method.

    References
    ----------
    Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing
    Financial Risk* (3rd ed.). McGraw-Hill.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    if len(r) < 2:
        raise ValueError("Need at least 2 return observations.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    method = method.lower()
    if method == "historical":
        var = -float(np.percentile(r, 100 * alpha))
    elif method == "parametric":
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        z = _st.norm.ppf(alpha)
        var = -(mu + z * sigma)
    elif method in ("cornish-fisher", "cornish_fisher", "cf"):
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        z = _st.norm.ppf(alpha)
        s = float(_st.skew(r))
        k = float(_st.kurtosis(r))
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        var = -(mu + z_cf * sigma)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'historical', 'parametric', or 'cornish-fisher'.")

    return DescriptiveResult(
        name="ValueAtRisk",
        value=float(var),
        extra={
            "alpha": alpha,
            "method": method,
            "n_obs": len(r),
        },
    )


valfn = value_at_risk


def cheatsheet() -> str:
    return "value_at_risk({}) -> Value at Risk (VaR)."
