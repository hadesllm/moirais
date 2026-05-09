# moirais.fn — function file (hadesllm/moirais)
"""AIC for model comparison."""

from __future__ import annotations

from moirais.fn._containers import ESRes


def cfa_aic(
    loglik: float,
    n_params: int,
) -> ESRes:
    """Akaike Information Criterion for a fitted model.

    AIC = -2 * loglik + 2 * n_params

    Parameters
    ----------
    loglik : float
        Log-likelihood of the model.
    n_params : int
        Number of estimated parameters.

    Returns
    -------
    ESRes
        measure="AIC".

    References
    ----------
    Akaike, H. (1974). A new look at the statistical model identification.
    IEEE Transactions on Automatic Control, 19(6), 716-723.
    """
    aic = -2.0 * loglik + 2.0 * n_params

    return ESRes(
        measure="AIC",
        estimate=float(aic),
        extra={"loglik": float(loglik), "n_params": n_params},
    )


aic = cfa_aic


def cheatsheet() -> str:
    return "cfa_aic({}) -> AIC for model comparison."
