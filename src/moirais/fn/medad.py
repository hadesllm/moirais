# moirais.fn — function file (hadesllm/moirais)
"""Mediation analysis — Baron-Kenny with Sobel test."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def mediation_analysis(y, treatment, mediator, covariates=None, cdf=None):
    """
    Baron-Kenny mediation analysis with Sobel test.

    Estimates indirect effect (a*b), direct effect (c'), and total effect (c).

    :param y: (n,) outcome.
    :param treatment: (n,) treatment/exposure.
    :param mediator: (n,) mediator variable.
    :param covariates: (n, p) optional covariates.
    :return: DescriptiveResult with indirect, direct, total effects and Sobel p-value.

    References
    ----------
    Baron RM, Kenny DA (1986). The moderator-mediator variable distinction.
    J Personality Social Psychology 51(6):1173-1182.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    t = np.asarray(treatment, dtype=np.float64).ravel()
    m = np.asarray(mediator, dtype=np.float64).ravel()
    n = len(y)

    def _ols(dep, *indep):
        X = np.column_stack([np.ones(n)] + [np.asarray(v).ravel() for v in indep])
        if covariates is not None:
            X = np.column_stack([X, np.asarray(covariates)])
        beta = np.linalg.lstsq(X, dep, rcond=None)[0]
        resid = dep - X @ beta
        sigma2 = np.sum(resid**2) / (n - X.shape[1])
        se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
        return beta, se

    beta_c, _ = _ols(y, t)
    total = beta_c[1]
    beta_a, se_a = _ols(m, t)
    a, sa = beta_a[1], se_a[1]
    beta_b, se_b = _ols(y, t, m)
    b, sb = beta_b[2], se_b[2]
    direct = beta_b[1]
    indirect = a * b
    sobel_se = np.sqrt(a**2 * sb**2 + b**2 * sa**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0.0
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    return DescriptiveResult(
        name="mediation_analysis",
        value=float(indirect),
        extra={
            "indirect_effect": float(indirect),
            "direct_effect": float(direct),
            "total_effect": float(total),
            "a_path": float(a),
            "b_path": float(b),
            "sobel_z": float(sobel_z),
            "sobel_p": float(sobel_p),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "mediation_analysis({}) -> Mediation analysis — Baron-Kenny with Sobel test."
