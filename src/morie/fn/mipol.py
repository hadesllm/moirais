# morie.fn — function file (hadesllm/morie)
"""Multiple imputation via predictive mean matching."""

import numpy as np

from ._containers import DescriptiveResult


def mi_pool(estimates, variances):
    """
    Pool estimates from multiple imputed datasets (Rubin's rules).

    :param estimates: (m,) point estimates from m imputed datasets.
    :param variances: (m,) within-imputation variances.
    :return: DescriptiveResult with pooled estimate, total variance, fraction missing info.

    References
    ----------
    Rubin DB (1987). Multiple Imputation for Nonresponse in Surveys.
    Wiley.
    """
    estimates = np.asarray(estimates, dtype=np.float64).ravel()
    variances = np.asarray(variances, dtype=np.float64).ravel()
    m = len(estimates)
    if m < 2:
        raise ValueError("Need at least 2 imputed datasets")

    q_bar = estimates.mean()
    u_bar = variances.mean()
    b = np.var(estimates, ddof=1)
    total_var = u_bar + (1 + 1 / m) * b
    gamma = (1 + 1 / m) * b / total_var if total_var > 0 else 0.0
    df_old = (m - 1) / gamma**2 if gamma > 0 else np.inf
    se = np.sqrt(total_var)

    return DescriptiveResult(
        name="mi_pool",
        value=float(q_bar),
        extra={
            "pooled_estimate": float(q_bar),
            "total_variance": float(total_var),
            "se": float(se),
            "within_variance": float(u_bar),
            "between_variance": float(b),
            "frac_missing_info": float(gamma),
            "df": float(df_old),
            "m": m,
        },
    )


def cheatsheet() -> str:
    return "mi_pool({}) -> Multiple imputation via predictive mean matching."
