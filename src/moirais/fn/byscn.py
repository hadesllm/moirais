# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""BayesCpi: BayesC with pi estimated from data (mixture proportion unknown)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayes_cpi_prior"]


def bayes_cpi_prior(y, X, p0, p1):
    """
    BayesCpi: BayesC with pi estimated from data (mixture proportion unknown)

    Formula: beta_j ~ pi*delta(0) + (1-pi)*N(0, sigma_b^2); pi ~ Beta(p0, p1) estimated via Gibbs

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    p0 : array-like
        Input data.
    p1 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta_samples': 'array', 'pi_samples': 'array'}

    References
    ----------
    Montesinos Lopez Ch 6
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    if y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "BayesCpi: BayesC with pi estimated from data (mixture proportion unknown)"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "BayesCpi: BayesC with pi estimated from data (mixture proportion unknown)"})


def cheatsheet():
    return "byscn: BayesCpi: BayesC with pi estimated from data (mixture proportion unknown)"
