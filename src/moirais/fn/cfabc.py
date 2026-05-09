# moirais.fn — function file (hadesllm/moirais)
"""BIC for model comparison."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def cfa_bic(
    loglik: float,
    n_params: int,
    n_obs: int,
) -> ESRes:
    """Bayesian Information Criterion for a fitted model.

    BIC = -2 * loglik + log(n) * n_params

    Parameters
    ----------
    loglik : float
        Log-likelihood of the model.
    n_params : int
        Number of estimated parameters.
    n_obs : int
        Number of observations.

    Returns
    -------
    ESRes
        measure="BIC".

    References
    ----------
    Schwarz, G. (1978). Estimating the dimension of a model.
    The Annals of Statistics, 6(2), 461-464.
    """
    bic = -2.0 * loglik + np.log(n_obs) * n_params

    return ESRes(
        measure="BIC",
        estimate=float(bic),
        n=n_obs,
        extra={"loglik": float(loglik), "n_params": n_params},
    )


bic = cfa_bic


def cheatsheet() -> str:
    return "cfa_bic({}) -> BIC for model comparison."
