"""Time discovers truth. -- Seneca"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import TestResult


def simulation_gof(
    observed: np.ndarray | list,
    simulated: np.ndarray | list,
    *,
    test: str = "ks",
) -> TestResult:
    """Goodness-of-fit test comparing observed data to simulated output.

    Tests whether the observed and simulated samples come from the same
    distribution.

    Parameters
    ----------
    observed : array-like
        Observed (real) data.
    simulated : array-like
        Simulated data from a model.
    test : str
        ``"ks"`` for Kolmogorov-Smirnov, ``"ad"`` for Anderson-Darling
        (via permutation approximation), ``"cvm"`` for Cramer-von Mises.

    Returns
    -------
    TestResult
        Statistic and p-value for H0: distributions are the same.
    """
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    obs = obs[~np.isnan(obs)]
    sim = sim[~np.isnan(sim)]
    if len(obs) < 2 or len(sim) < 2:
        raise ValueError("Need at least 2 samples in each group")

    if test == "ks":
        stat, p = _st.ks_2samp(obs, sim)
        method = "Kolmogorov-Smirnov"
    elif test == "cvm":
        result = _st.cramervonmises_2samp(obs, sim)
        stat, p = result.statistic, result.pvalue
        method = "Cramer-von Mises"
    elif test == "ad":
        result = _st.anderson_ksamp([obs, sim])
        stat = result.statistic
        p = result.pvalue
        method = "Anderson-Darling k-sample"
    else:
        raise ValueError(f"test must be 'ks', 'ad', or 'cvm', got '{test}'")

    return TestResult(
        test_name="Simulation GoF",
        statistic=float(stat),
        p_value=float(p),
        method=method,
        n=len(obs) + len(sim),
        extra={"n_obs": len(obs), "n_sim": len(sim)},
    )


simul = simulation_gof


def cheatsheet() -> str:
    return "simulation_gof({}) -> Goodness of fit to simulation. 'What is real? How do you def"
