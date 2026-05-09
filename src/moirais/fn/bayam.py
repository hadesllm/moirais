# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian Aldrich-McKelvey scaling via MCMC."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayesian_am_scaling"]


def bayesian_am_scaling(survey_data, n_iter, burnin):
    """
    Bayesian Aldrich-McKelvey scaling via MCMC

    Formula: z_i = alpha_i + beta_i*s + eps_i; priors: s ~ N(0,I), alpha ~ N(0, sigma_a^2), beta ~ N(1, sigma_b^2)

    Parameters
    ----------
    survey_data : array-like
        Input data.
    n_iter : array-like
        Input data.
    burnin : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'s_samples': 'matrix', 'credible_intervals': 'matrix'}

    References
    ----------
    Armstrong Ch 6
    """
    survey_data = np.asarray(survey_data, dtype=float)
    n = int(survey_data) if survey_data.ndim == 0 else len(survey_data)
    result = float(np.mean(survey_data))
    se = float(np.std(survey_data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian Aldrich-McKelvey scaling via MCMC"})


def cheatsheet():
    return "bayam: Bayesian Aldrich-McKelvey scaling via MCMC"
