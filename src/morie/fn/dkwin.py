# morie.fn -- function file (hadesllm/morie)
"""Dvoretzky-Kiefer-Wolfowitz inequality for empirical CDF."""

from __future__ import annotations

import numpy as np
from scipy import stats


def dkw_test(
    x: np.ndarray,
    *,
    cdf: str = "norm",
    alpha: float = 0.05,
    cdf_params: tuple = (),
) -> dict:
    r"""
    DKW inequality-based goodness-of-fit test for an empirical CDF.

    The DKW inequality provides a distribution-free bound on the
    Kolmogorov-Smirnov statistic :math:`D_n = \sup_t |\hat{F}_n(t) - F_0(t)|`:

    .. math::

        P(D_n > \varepsilon) \le 2\exp(-2n\varepsilon^2)

    The critical value at level :math:`\alpha` is:

    .. math::

        \varepsilon_\alpha = \sqrt{\frac{\ln(2/\alpha)}{2n}}

    :param x: 1-D array of observations.
    :param cdf: Null distribution name (scipy.stats distribution). Default ``"norm"``.
    :param alpha: Significance level. Default 0.05.
    :param cdf_params: Extra parameters for the scipy distribution.
    :return: dict with ``D_n`` (KS statistic), ``critical_value``,
        ``dkw_bound`` (probability bound at D_n), ``reject``, ``n``.
    :raises ValueError: If x is empty, alpha not in (0,1), or cdf unknown.

    References
    ----------
    Dvoretzky, A., Kiefer, J., & Wolfowitz, J. (1956). Asymptotic minimax
        character of the sample distribution function. *Ann. Math. Statist.*,
        27(3), 642--669.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Ch. 2. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    try:
        dist = getattr(stats, cdf)
    except AttributeError:
        raise ValueError(f"Unknown distribution: {cdf}.")

    n = x.size
    x_sorted = np.sort(x)
    ecdf_vals = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    F0 = dist.cdf(x_sorted, *cdf_params)

    D_plus = np.max(ecdf_vals - F0)
    D_minus = np.max(F0 - ecdf_prev)
    D_n = max(D_plus, D_minus)

    critical_value = np.sqrt(np.log(2.0 / alpha) / (2.0 * n))
    dkw_bound = 2.0 * np.exp(-2.0 * n * D_n**2)
    dkw_bound = min(dkw_bound, 1.0)

    return {
        "D_n": float(D_n),
        "critical_value": float(critical_value),
        "dkw_bound": float(dkw_bound),
        "reject": bool(D_n > critical_value),
        "n": n,
        "alpha": float(alpha),
        "cdf": cdf,
        "method": "DKW inequality test",
    }


dkwin = dkw_test


def cheatsheet() -> str:
    return "dkw_test({x}) -> DKW inequality-based goodness-of-fit test."
