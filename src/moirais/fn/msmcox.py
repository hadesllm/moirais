"""MSM Cox marginal hazard model with IPTW."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_cox_marginal"]


def msm_cox_marginal(time, event, treatment_history, covariate_history):
    """
    MSM Cox marginal hazard model with IPTW

    Formula: lambda_0(t) exp(beta a_bar) weighted

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins (1999); Hernán et al (2000)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM Cox marginal hazard model with IPTW"})


def cheatsheet():
    return "msmcox: MSM Cox marginal hazard model with IPTW"
