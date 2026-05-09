# moirais.fn — function file (hadesllm/moirais)
"""
hzrat.py - Hazard ratio estimation with confidence interval.

Estimates the hazard ratio between two groups using a Cox model (partial
likelihood), with Wald-type confidence interval and p-value.

Reference: Cox, D.R. (1972). Regression models and life-tables. Journal of the
Royal Statistical Society, Series B, 34(2), 187-220.
"""

__all__ = ["hzrat"]

import numpy as np


def hzrat(
    time: np.ndarray,
    event: np.ndarray,
    group: np.ndarray,
    reference: object = None,
    alpha: float = 0.05,
    ties: str = "breslow",
) -> dict:
    """
    Estimate hazard ratio between groups from a two-sample Cox model.

    For a binary group indicator, this fits the Cox model:
        h(t | group) = h_0(t) * exp(beta * group)
    and returns HR = exp(beta) with 95% CI.

    For k > 2 groups, one group is the reference and HR is estimated for
    each group versus the reference.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    group : np.ndarray, shape (n,)
        Group labels. Reference group defaults to the first sorted value.
    reference : object, optional
        Reference group label. Default is the smallest sorted label.
    alpha : float, optional
        Significance level for CI. Default 0.05.
    ties : str, optional
        Tie-handling: 'breslow' (default) or 'efron'.

    Returns
    -------
    dict
        hazard_ratio : float (or dict for k>2 groups)
            Estimated hazard ratio(s).
        ci_lower : float (or dict)
            Lower confidence bound.
        ci_upper : float (or dict)
            Upper confidence bound.
        p_value : float (or dict)
            Two-sided Wald p-value.
        beta : float (or np.ndarray)
            Log hazard ratio coefficient(s).
        se : float (or np.ndarray)
            Standard error(s) of beta.
        log_likelihood : float
            Partial log-likelihood.

    Raises
    ------
    ValueError
        If fewer than 2 groups or inputs are invalid.

    References
    ----------
    Cox, D.R. (1972). Journal of the Royal Statistical Society, Series B,
    34(2), 187-220.
    """
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    group = np.asarray(group)
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")

    groups = sorted(np.unique(group).tolist())
    if len(groups) < 2:
        raise ValueError("At least two groups required.")

    if reference is None:
        reference = groups[0]
    elif reference not in groups:
        raise ValueError(f"Reference group '{reference}' not found in group array.")

    # Create dummy variables (reference group = 0)
    non_ref_groups = [g for g in groups if g != reference]
    X = np.column_stack([(group == g).astype(float) for g in non_ref_groups])

    # Fit Cox model via partial likelihood
    from moirais.fn.cxphr import cxphr
    result = cxphr(time, event, X, ties=ties)

    beta = result["beta"]
    se = result["se"]
    z = result["z"]
    p_val = result["p_value"]
    ll = result["log_likelihood"]

    z_alpha = float(_stats.norm.ppf(1 - alpha / 2))
    hr = np.exp(beta)
    ci_lo = np.exp(beta - z_alpha * se)
    ci_hi = np.exp(beta + z_alpha * se)

    if len(non_ref_groups) == 1:
        return {
            "hazard_ratio": float(hr[0]),
            "ci_lower": float(ci_lo[0]),
            "ci_upper": float(ci_hi[0]),
            "p_value": float(p_val[0]),
            "beta": float(beta[0]),
            "se": float(se[0]),
            "log_likelihood": float(ll),
            "reference_group": reference,
            "comparison_group": non_ref_groups[0],
        }
    else:
        return {
            "hazard_ratio": {str(g): float(hr[i]) for i, g in enumerate(non_ref_groups)},
            "ci_lower": {str(g): float(ci_lo[i]) for i, g in enumerate(non_ref_groups)},
            "ci_upper": {str(g): float(ci_hi[i]) for i, g in enumerate(non_ref_groups)},
            "p_value": {str(g): float(p_val[i]) for i, g in enumerate(non_ref_groups)},
            "beta": beta,
            "se": se,
            "log_likelihood": float(ll),
            "reference_group": reference,
            "comparison_groups": non_ref_groups,
        }
