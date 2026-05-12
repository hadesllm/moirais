# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Age-period-cohort (APC) decomposition via intrinsic estimator."""

import numpy as np

from ._containers import DescriptiveResult


def age_period_cohort(
    rates: np.ndarray,
    age_groups: np.ndarray | None = None,
    periods: np.ndarray | None = None,
) -> DescriptiveResult:
    """Age-period-cohort decomposition using the intrinsic estimator.

    The intrinsic estimator (IE) resolves the linear dependency in APC
    models by projecting coefficients onto the orthogonal complement of
    the null space of the design matrix.

    Parameters
    ----------
    rates : array-like, shape (n_age, n_period)
        Matrix of rates (age groups as rows, periods as columns).
    age_groups : array-like or None
        Labels for age groups. If None, uses 0-indexed integers.
    periods : array-like or None
        Labels for periods. If None, uses 0-indexed integers.

    Returns
    -------
    DescriptiveResult
        value = intercept. extra contains age_effects, period_effects,
        cohort_effects arrays.

    References
    ----------
    Yang, Y., Fu, W. J., & Land, K. C. (2004). A methodological
    comparison of age-period-cohort models: the intrinsic estimator and
    conventional generalized linear models. Sociological Methodology,
    34(1), 75-110.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2:
        raise ValueError("rates must be a 2D array (age x period)")

    n_age, n_period = rates.shape
    n_cohort = n_age + n_period - 1

    if age_groups is None:
        age_groups = np.arange(n_age)
    if periods is None:
        periods = np.arange(n_period)

    age_groups = np.asarray(age_groups)
    periods = np.asarray(periods)

    y = np.log(rates + 1e-10).ravel()
    n = len(y)

    X_age = np.zeros((n, n_age))
    X_per = np.zeros((n, n_period))
    X_coh = np.zeros((n, n_cohort))

    for i in range(n_age):
        for j in range(n_period):
            idx = i * n_period + j
            X_age[idx, i] = 1
            X_per[idx, j] = 1
            c = i - j + n_period - 1
            X_coh[idx, c] = 1

    X = np.column_stack([np.ones(n), X_age[:, :-1], X_per[:, :-1], X_coh[:, :-1]])

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    tol = 1e-10 * s[0]
    rank = np.sum(s > tol)

    s_inv = np.zeros_like(s)
    s_inv[:rank] = 1.0 / s[:rank]

    beta_ie = Vt.T @ np.diag(s_inv) @ U.T @ y

    intercept = beta_ie[0]
    a_eff = np.zeros(n_age)
    a_eff[:-1] = beta_ie[1:n_age]
    a_eff[-1] = -np.sum(a_eff[:-1])

    p_eff = np.zeros(n_period)
    p_eff[:-1] = beta_ie[n_age : n_age + n_period - 1]
    p_eff[-1] = -np.sum(p_eff[:-1])

    c_eff = np.zeros(n_cohort)
    c_eff[:-1] = beta_ie[n_age + n_period - 1 :]
    c_eff[-1] = -np.sum(c_eff[:-1])

    return DescriptiveResult(
        name="APC decomposition",
        value=float(intercept),
        extra={
            "age_effects": a_eff,
            "period_effects": p_eff,
            "cohort_effects": c_eff,
            "age_groups": age_groups,
            "periods": periods,
        },
    )


apc = age_period_cohort


def cheatsheet() -> str:
    return "age_period_cohort({}) -> Age-period-cohort (APC) decomposition via intrinsic estimato"
